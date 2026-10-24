import json
from pathlib import Path

notebook_path = Path(r"c:\Users\ACER\Desktop\HackatonBG\notebooks\BG_Copiloto_Entrenamiento_Colab.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

new_viz_code = [
    "# =========================================================================================\n",
    "# DASHBOARD VISUAL MULTI-MÉTRICA PARA HACKATHON BANCO GUAYAQUIL\n",
    "# =========================================================================================\n",
    "from sklearn.metrics import confusion_matrix\n",
    "\n",
    "fig, axes = plt.subplots(2, 2, figsize=(18, 12))\n",
    "\n",
    "# --- PANEL 1: COMPARATIVA DE MÉTRICAS CLAVE (F1, RECALL DE MORA Y ACCURACY) ---\n",
    "model_names_short = [m.split('.')[1].split('(')[0].strip() for m in df_benchmark['Modelo / Técnica']]\n",
    "x_pos = np.arange(len(model_names_short))\n",
    "bar_width = 0.25\n",
    "\n",
    "axes[0, 0].bar(x_pos - bar_width, df_benchmark['F1-Score'], width=bar_width, label='F1-Score', color='#1f77b4')\n",
    "axes[0, 0].bar(x_pos, df_benchmark['Recall (Mora)'], width=bar_width, label='Recall de Mora (Sensibilidad)', color='#2ca02c')\n",
    "axes[0, 0].bar(x_pos + bar_width, df_benchmark['Accuracy (50/50)'], width=bar_width, label='Accuracy Global', color='#ff7f0e')\n",
    "axes[0, 0].set_xticks(x_pos)\n",
    "axes[0, 0].set_xticklabels(model_names_short, rotation=20, ha='right', fontsize=10)\n",
    "axes[0, 0].set_ylim(0.90, 1.00)\n",
    "axes[0, 0].set_title('1. Rendimiento Predictivo por Modelo (F1, Recall, Accuracy)', fontsize=12, fontweight='bold')\n",
    "axes[0, 0].set_ylabel('Puntaje (0.90 a 1.00)')\n",
    "axes[0, 0].legend(loc='lower right')\n",
    "axes[0, 0].grid(axis='y', alpha=0.3)\n",
    "\n",
    "# --- PANEL 2: EFICIENCIA COMPUTACIONAL (TIEMPO DE ENTRENAMIENTO EN SEGUNDOS) ---\n",
    "time_colors = ['#d9534f' if t > 10 else '#2e7d32' for t in df_benchmark['Tiempo (s)']]\n",
    "bars_time = axes[0, 1].barh(model_names_short, df_benchmark['Tiempo (s)'], color=time_colors)\n",
    "axes[0, 1].set_title('2. Velocidad de Entrenamiento (Segundos en Google Colab)', fontsize=12, fontweight='bold')\n",
    "axes[0, 1].set_xlabel('Tiempo en segundos (Menor es mejor para producción)')\n",
    "for bar in bars_time:\n",
    "    w = bar.get_width()\n",
    "    axes[0, 1].text(w + 0.5, bar.get_y() + bar.get_height() / 2, f'{w:.2f}s', va='center', fontsize=10, fontweight='bold')\n",
    "axes[0, 1].grid(axis='x', alpha=0.3)\n",
    "\n",
    "# --- PANEL 3: MATRIZ DE CONFUSIÓN NORMALIZADA DEL MODELO GANADOR (LightGBM) ---\n",
    "best_model = candidate_models['6. LightGBM (Leaf-wise Boosting)']\n",
    "y_pred_best = best_model.predict(X_test)\n",
    "cm = confusion_matrix(y_test, y_pred_best, normalize='true') * 100\n",
    "sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', cbar=False, ax=axes[1, 0],\n",
    "            xticklabels=['0: Saludable', '1: Riesgo Mora'], yticklabels=['0: Saludable', '1: Riesgo Mora'], annot_kws={'size': 13, 'weight': 'bold'})\n",
    "axes[1, 0].set_title('3. Matriz de Confusión (% Normalizado) — LightGBM (Ganador)', fontsize=12, fontweight='bold')\n",
    "axes[1, 0].set_xlabel('Predicción del Modelo de Banco Guayaquil')\n",
    "axes[1, 0].set_ylabel('Estado Real del Cliente')\n",
    "\n",
    "# --- PANEL 4: IMPORTANCIA RELATIVA DE VARIABLES (%) ---\n",
    "avg_importance = (\n",
    "    feature_importances_dict['5. XGBoost (Depth-wise Boosting)']\n",
    "    + feature_importances_dict['6. LightGBM (Leaf-wise Boosting)']\n",
    ") / 2.0\n",
    "avg_importance = avg_importance.sort_values(ascending=True)\n",
    "colors = ['#d9534f' if col == 'SCORE_ESTILO_VIDA_REDES' else '#1f4e78' for col in avg_importance.index]\n",
    "axes[1, 1].barh(avg_importance.index, avg_importance.values, color=colors)\n",
    "axes[1, 1].axvline(5.0, color='#d9534f', linestyle='--', lw=1.5, label='Límite ético Redes Sociales (5%)')\n",
    "axes[1, 1].set_title('4. Importancia de Variables en la Decisión (%) — Top Boosters', fontsize=12, fontweight='bold')\n",
    "axes[1, 1].set_xlabel('Peso Porcentual en la Decisión (%)')\n",
    "axes[1, 1].legend(loc='lower right')\n",
    "axes[1, 1].grid(axis='x', alpha=0.3)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]

nb["cells"][-1]["source"] = new_viz_code

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Updated visualization cell successfully!")

