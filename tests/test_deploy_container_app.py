"""Exercise releases against a fake Azure CLI; no Azure resources are changed."""

import copy
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest


GIT_BASH = Path("C:/Program Files/Git/bin/bash.exe")
BASH = str(GIT_BASH) if GIT_BASH.exists() else shutil.which("bash")
JQ = os.environ.get("JQ_TEST_BINARY") or shutil.which("jq")
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/deploy-container-app.sh"
PRINCIPAL = "c20c5821-e871-473b-a806-464dae12b7c6"
IDENTITY = "/subscriptions/test/resourceGroups/rg-test/providers/Microsoft.ManagedIdentity/userAssignedIdentities/id-credit-bg-dev"
IMAGE = "registry.azurecr.io/ia-credit-bg@sha256:" + "a" * 64
REVISION = "ca-ia-credit-bg-dev--b292-a1"

MOCK = r'''
import json
import os
from pathlib import Path
import sys

mode, *args = sys.argv[1:]
root = Path(os.environ['MOCK_ROOT'])
fixture = json.loads((root / 'fixture.json').read_text())
with (root / 'calls.jsonl').open('a') as handle:
    handle.write(json.dumps([mode, *args]) + '\n')
if mode == 'curl':
    if args[-1].endswith('/healthz'):
        print(json.dumps({'status': 'DOWN' if fixture.get('bad_health') else 'UP'}))
    else:
        print(json.dumps({'paths': {} if fixture.get('bad_openapi') else {'/api/v1/copiloto/evaluar': {'post': {}}}}))
elif args[:2] == ['extension', 'show']:
    pass
elif args[:2] == ['containerapp', 'show']:
    if fixture.get('missing_app'):
        sys.exit(3)
    counter = root / 'shows.txt'
    count = int(counter.read_text()) if counter.exists() else 0
    counter.write_text(str(count + 1))
    if count == 0:
        print(json.dumps(fixture['app']))
    else:
        state = 'Failed' if fixture.get('failed_revision') else 'Succeeded'
        ready = None if fixture.get('never_ready') else os.environ['TEST_REVISION']
        print(json.dumps({'state': state, 'latestRevision': os.environ['TEST_REVISION'], 'latestReadyRevision': ready}))
elif args[:2] == ['acr', 'show']:
    print(json.dumps({'id': '/subscriptions/test/resourceGroups/rg-test/providers/Microsoft.ContainerRegistry/registries/registry', 'loginServer': 'registry.azurecr.io', 'roleAssignmentMode': fixture.get('permission_mode', 'LegacyRegistryPermissions')}))
elif args[:4] == ['acr', 'config', 'authentication-as-arm', 'show']:
    print(fixture.get('arm_auth', 'enabled'))
elif args[:3] == ['role', 'assignment', 'list']:
    assert args[args.index('--assignee') + 1] == os.environ['TEST_PRINCIPAL']
    assert args[args.index('--fill-principal-name') + 1] == 'false'
    assert '--include-inherited' in args
    print(json.dumps([] if fixture.get('no_acrpull') else [{'roleDefinitionId': '/subscriptions/test/providers/Microsoft.Authorization/roleDefinitions/7f951dda-4ed3-4680-a7ca-43fe172d538d'}]))
elif args[:2] == ['containerapp', 'update']:
    sys.exit(9 if fixture.get('update_error') else 0)
elif args[:3] == ['containerapp', 'revision', 'show']:
    print(json.dumps({'name': os.environ['TEST_REVISION'], 'containers': [{'name': 'ia-credit-bg', 'image': 'wrong-image' if fixture.get('wrong_image') else os.environ['TEST_IMAGE']}]}))
else:
    print('Unexpected mutation or command: ' + repr(args), file=sys.stderr)
    sys.exit(99)
'''


@unittest.skipUnless(BASH and JQ, "Bash and jq are required for release checks")
class ContainerAppReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        (self.root / "mock.py").write_text(MOCK, encoding="utf-8")
        python = shlex.quote(sys.executable.replace("\\", "/"))
        mock = shlex.quote(str(self.root / "mock.py").replace("\\", "/"))
        for name in ("az", "curl"):
            self.executable(name, f'#!/usr/bin/env bash\nexec {python} {mock} {name} "$@"\n')
        jq = shlex.quote(str(JQ).replace("\\", "/"))
        binary = "--binary " if os.name == "nt" else ""
        self.executable("jq", f'#!/usr/bin/env bash\nexec {jq} {binary}"$@"\n')
        self.executable("sleep", '#!/usr/bin/env bash\nexec /usr/bin/sleep 0.01\n')
        self.app = {
            "id": "/subscriptions/test/resourceGroups/rg-test/providers/Microsoft.App/containerApps/ca-ia-credit-bg-dev",
            "name": "ca-ia-credit-bg-dev",
            "identity": {"userAssignedIdentities": {IDENTITY.lower(): {"principalId": PRINCIPAL}}},
            "mode": "Single",
            "registries": [{"server": "registry.azurecr.io", "identity": IDENTITY}],
            "ingress": {"external": True, "allowInsecure": False, "targetPort": 8000, "fqdn": "test.azurecontainerapps.io", "traffic": [{"latestRevision": True, "weight": 100}]},
            "containers": [{"name": "ia-credit-bg", "image": "registry.azurecr.io/ia-credit-bg:257"}],
            "latestRevision": "ca-ia-credit-bg-dev--terraform",
        }

    def executable(self, name, source):
        path = self.bin / name
        path.write_text(source, encoding="utf-8", newline="\n")
        path.chmod(0o755)

    def release(self, fixture=None, overrides=None):
        data = {"app": copy.deepcopy(self.app)}
        data.update(fixture or {})
        (self.root / "fixture.json").write_text(json.dumps(data), encoding="utf-8")
        (self.root / "calls.jsonl").write_text("", encoding="utf-8")
        (self.root / "shows.txt").write_text("0", encoding="utf-8")
        env = dict(os.environ)
        env.update({
            "PATH": str(self.bin) + os.pathsep + env["PATH"],
            "MSYS_NO_PATHCONV": "1",
            "MOCK_ROOT": str(self.root), "TEST_IMAGE": IMAGE,
            "MOCK_BIN": str(self.bin).replace("\\", "/"),
            "RELEASE_SCRIPT": str(SCRIPT).replace("\\", "/"),
            "TEST_PRINCIPAL": PRINCIPAL, "TEST_REVISION": REVISION,
            "RESOURCE_GROUP": "rg-test", "CONTAINER_APP_NAME": "ca-ia-credit-bg-dev",
            "CONTAINER_NAME": "ia-credit-bg", "ACR_NAME": "registry",
            "IMAGE_REPOSITORY": "ia-credit-bg", "CONTAINER_IMAGE": IMAGE,
            "REVISION_SUFFIX": "b292-a1", "DEPLOYMENT_TIMEOUT_SECONDS": "2",
            "DEPLOY_REPORT_DIR": str(self.root / "reports"),
        })
        env.update(overrides or {})
        # Git Bash prepends its own binaries on startup; set mocks afterwards.
        prefix = '$(cygpath -u "$MOCK_BIN")' if os.name == "nt" else "$MOCK_BIN"
        command = f'export PATH="{prefix}:$PATH"; source "$RELEASE_SCRIPT"'
        result = subprocess.run([BASH, "-c", command], env=env, text=True, capture_output=True, timeout=40)
        self.calls = [json.loads(line) for line in (self.root / "calls.jsonl").read_text().splitlines()]
        return result

    def assert_no_update(self):
        self.assertFalse(any(call[:3] == ["az", "containerapp", "update"] for call in self.calls))

    def test_updates_only_named_container_image_and_revision(self):
        result = self.release()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        updates = [call for call in self.calls if call[:3] == ["az", "containerapp", "update"]]
        self.assertEqual(updates, [["az", "containerapp", "update", "--name", "ca-ia-credit-bg-dev", "--resource-group", "rg-test", "--container-name", "ia-credit-bg", "--image", IMAGE, "--revision-suffix", "b292-a1", "--no-wait", "--output", "none"]])
        self.assertTrue((self.root / "reports/summary.md").exists())

    def test_resumes_a_revision_already_submitted(self):
        self.app["latestRevision"] = REVISION
        self.app["containers"][0]["image"] = IMAGE
        result = self.release()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assert_no_update()

    def test_rejects_missing_resources_or_permissions_before_update(self):
        for fixture in ({"missing_app": True}, {"no_acrpull": True}, {"permission_mode": "AbacRepositoryPermissions"}, {"arm_auth": "disabled"}):
            with self.subTest(fixture=fixture):
                self.assertNotEqual(self.release(fixture).returncode, 0)
                self.assert_no_update()

    def test_rejects_incompatible_terraform_configuration(self):
        for field, value in (("containers", []), ("mode", "Multiple"), ("registries", []), ("identity", {"userAssignedIdentities": {}})):
            with self.subTest(field=field):
                app = copy.deepcopy(self.app)
                app[field] = value
                self.assertNotEqual(self.release({"app": app}).returncode, 0)
                self.assert_no_update()

    def test_rejects_unpinned_or_foreign_images(self):
        for image in ("registry.azurecr.io/ia-credit-bg:latest", "other.azurecr.io/ia-credit-bg@sha256:" + "a" * 64):
            with self.subTest(image=image):
                self.assertNotEqual(self.release(overrides={"CONTAINER_IMAGE": image}).returncode, 0)
                self.assert_no_update()

    def test_propagates_update_failure(self):
        self.assertEqual(self.release({"update_error": True}).returncode, 9)

    def test_rejects_failed_or_incorrect_revisions(self):
        for fixture in ({"failed_revision": True}, {"wrong_image": True}, {"never_ready": True}):
            with self.subTest(fixture=fixture):
                self.assertNotEqual(self.release(fixture).returncode, 0)

    def test_requires_healthy_api_and_expected_contract(self):
        for fixture in ({"bad_health": True}, {"bad_openapi": True}):
            with self.subTest(fixture=fixture):
                self.assertNotEqual(self.release(fixture).returncode, 0)

    def test_rejects_invalid_revision_suffix(self):
        self.assertNotEqual(self.release(overrides={"REVISION_SUFFIX": "invalid--suffix"}).returncode, 0)
        self.assert_no_update()


if __name__ == "__main__":
    unittest.main()
