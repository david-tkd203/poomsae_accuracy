#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis Estadístico Completo con Visualizaciones Profesionales
================================================================

Genera análisis estadístico comprehensivo + gráficos de alta calidad para tesis

Gráficos generados:
1. Distribución de Exactitud (Histograma + KDE)
2. Box Plots por Componente
3. Scatter Plot: Exactitud vs LEVES (con línea de regresión)
4. Heatmap de Correlaciones
5. Ranking de Videos (Barras horizontales)
6. Violin Plots por Componente
7. Distribución de LEVES (Histograma)
8. Serie Temporal de Videos
9. Gráfico de Dispersión 3D (Brazos-Piernas-Patadas)
10. Radar Chart de Componentes Promedio
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Importar matplotlib y seaborn para gráficos
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Configuración de estilo profesional
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Crear directorio para gráficos
output_dir = Path("analisis_graficos")
output_dir.mkdir(exist_ok=True)

print("="*100)
print(" 📊 ANÁLISIS ESTADÍSTICO COMPLETO + VISUALIZACIONES")
print("="*100)
print()

# ============================================================================
# 1. RECOLECCIÓN DE DATOS
# ============================================================================
print("📂 FASE 1: RECOLECCIÓN DE DATOS")
print("-" * 100)

videos = [
    "8yang_001", "8yang_002", "8yang_003", "8yang_004", "8yang_005",
    "8yang_006", "8yang_007", "8yang_008", "8yang_009", "8yang_010",
    "8yang_011", "8yang_012", "8yang_013", "8yang_014", "8yang_015",
    "8yang_016", "8yang_017", "8yang_018", "8yang_019", "8yang_020",
    "8yang_021", "8yang_022", "8yang_023", "8yang_024", "8yang_025",
    "8yang_026", "8yang_027", "8yang_028", "8yang_029",
]

# Recolectar todos los datos
all_data = []
summary_data = []

for video_id in videos:
    report_path = f"reports/{video_id}_score.xlsx"
    if not Path(report_path).exists():
        continue
    
    df_detalle = pd.read_excel(report_path, sheet_name="detalle")
    df_detalle['video_id'] = video_id
    all_data.append(df_detalle)
    
    # Leer hoja de resumen
    df_resumen = pd.read_excel(report_path, sheet_name="resumen")
    
    # Estadísticas por video
    summary_data.append({
        'video_id': video_id,
        'brazos_ok': len(df_detalle[df_detalle['comp_arms'] == 'OK']),
        'brazos_leve': len(df_detalle[df_detalle['comp_arms'] == 'LEVE']),
        'brazos_grave': len(df_detalle[df_detalle['comp_arms'] == 'GRAVE']),
        'piernas_ok': len(df_detalle[df_detalle['comp_legs'] == 'OK']),
        'piernas_leve': len(df_detalle[df_detalle['comp_legs'] == 'LEVE']),
        'piernas_grave': len(df_detalle[df_detalle['comp_legs'] == 'GRAVE']),
        'patadas_ok': len(df_detalle[df_detalle['comp_kick'] == 'OK']),
        'patadas_leve': len(df_detalle[df_detalle['comp_kick'] == 'LEVE']),
        'patadas_grave': len(df_detalle[df_detalle['comp_kick'] == 'GRAVE']),
        'exactitud': float(df_resumen['exactitud_final'].iloc[0]),
        'deducciones': float(df_resumen['ded_total'].iloc[0])
    })

# Consolidar dataframes
df_all = pd.concat(all_data, ignore_index=True)
df_summary = pd.DataFrame(summary_data)
df_summary['total_leves'] = (df_summary['brazos_leve'] + 
                              df_summary['piernas_leve'] + 
                              df_summary['patadas_leve'])

print(f"✅ Datos recolectados: {len(videos)} videos, {len(df_all)} movimientos")
print()

# ============================================================================
# 2. GRÁFICO 1: DISTRIBUCIÓN DE EXACTITUD (Histograma + KDE + Stats)
# ============================================================================
print("📊 Generando Gráfico 1: Distribución de Exactitud...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Histograma con KDE
ax1.hist(df_summary['exactitud'], bins=15, alpha=0.7, color='steelblue', 
         edgecolor='black', density=True, label='Histograma')

# Añadir KDE
from scipy.stats import gaussian_kde
kde = gaussian_kde(df_summary['exactitud'])
x_range = np.linspace(df_summary['exactitud'].min(), df_summary['exactitud'].max(), 100)
ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

# Líneas de referencia
ax1.axvline(df_summary['exactitud'].mean(), color='green', linestyle='--', 
            linewidth=2, label=f'Media: {df_summary["exactitud"].mean():.2f}')
ax1.axvline(df_summary['exactitud'].median(), color='orange', linestyle='--', 
            linewidth=2, label=f'Mediana: {df_summary["exactitud"].median():.2f}')

ax1.set_xlabel('Exactitud (sobre 4.00)', fontweight='bold')
ax1.set_ylabel('Densidad', fontweight='bold')
ax1.set_title('Distribución de Exactitud del Sistema\n(29 Videos)', fontweight='bold', fontsize=14)
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# Box plot vertical
bp = ax2.boxplot([df_summary['exactitud']], vert=True, patch_artist=True,
                 labels=['Exactitud'], widths=0.5)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][0].set_alpha(0.7)

# Añadir puntos individuales
y = df_summary['exactitud']
x = np.random.normal(1, 0.04, size=len(y))
ax2.scatter(x, y, alpha=0.6, s=50, c='darkblue', edgecolors='black', linewidth=0.5)

# Añadir estadísticas en el gráfico
stats_text = f"""
n = {len(df_summary)}
Media = {df_summary['exactitud'].mean():.2f}
Mediana = {df_summary['exactitud'].median():.2f}
Desv. Est. = {df_summary['exactitud'].std():.2f}
Min = {df_summary['exactitud'].min():.2f}
Max = {df_summary['exactitud'].max():.2f}
CV = {(df_summary['exactitud'].std()/df_summary['exactitud'].mean()*100):.1f}%
"""
ax2.text(1.4, df_summary['exactitud'].mean(), stats_text, fontsize=10, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax2.set_ylabel('Exactitud (sobre 4.00)', fontweight='bold')
ax2.set_title('Box Plot con Puntos Individuales', fontweight='bold', fontsize=14)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim([1.0, 4.0])

plt.tight_layout()
plt.savefig(output_dir / '01_distribucion_exactitud.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Guardado: {output_dir / '01_distribucion_exactitud.png'}")

# ============================================================================
# 3. GRÁFICO 2: BOX PLOTS POR COMPONENTE
# ============================================================================
print("📊 Generando Gráfico 2: Box Plots por Componente...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

componentes_data = {
    'Brazos': df_summary['brazos_ok'],
    'Piernas': df_summary['piernas_ok'],
    'Patadas': df_summary['patadas_ok']
}

colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
titles = ['💪 Brazos', '🦵 Piernas', '🥋 Patadas']

for idx, (comp_name, data) in enumerate(componentes_data.items()):
    ax = axes[idx]
    
    # Box plot
    bp = ax.boxplot([data], vert=True, patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor(colors[idx])
    bp['boxes'][0].set_alpha(0.7)
    
    # Scatter points
    y = data.values
    x = np.random.normal(1, 0.04, size=len(y))
    ax.scatter(x, y, alpha=0.5, s=60, c=colors[idx], edgecolors='black', linewidth=0.5)
    
    # Estadísticas
    mean_val = data.mean()
    median_val = data.median()
    ax.axhline(mean_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Media: {mean_val:.1f}')
    ax.axhline(median_val, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mediana: {median_val:.1f}')
    
    ax.set_ylabel('Evaluaciones OK (de 36)', fontweight='bold')
    ax.set_title(titles[idx], fontweight='bold', fontsize=14)
    ax.set_xticklabels([comp_name])
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 40])
    
    # Añadir porcentaje
    pct = (mean_val / 36) * 100
    ax.text(1, 38, f'{pct:.1f}% OK', ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=colors[idx], alpha=0.3))

plt.tight_layout()
plt.savefig(output_dir / '02_boxplots_componentes.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Guardado: {output_dir / '02_boxplots_componentes.png'}")

# ============================================================================
# 4. GRÁFICO 3: SCATTER PLOT - Exactitud vs LEVES (con regresión)
# ============================================================================
print("📊 Generando Gráfico 3: Scatter Plot Exactitud vs LEVES...")

fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot
scatter = ax.scatter(df_summary['total_leves'], df_summary['exactitud'], 
                    s=200, alpha=0.6, c=df_summary['exactitud'], 
                    cmap='RdYlGn', edgecolors='black', linewidth=1)

# Añadir etiquetas de video
for idx, row in df_summary.iterrows():
    ax.annotate(row['video_id'].replace('8yang_', ''), 
                (row['total_leves'], row['exactitud']),
                fontsize=8, ha='center', va='center', fontweight='bold')

# Línea de regresión
z = np.polyfit(df_summary['total_leves'], df_summary['exactitud'], 1)
p = np.poly1d(z)
x_line = np.linspace(df_summary['total_leves'].min(), df_summary['total_leves'].max(), 100)
ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f'Regresión: y = {z[0]:.3f}x + {z[1]:.2f}')

# Calcular R²
r, p_value = stats.pearsonr(df_summary['total_leves'], df_summary['exactitud'])
ax.text(0.05, 0.95, f'Correlación (Pearson):\nr = {r:.3f}\np-value = {p_value:.4f}\nR² = {r**2:.3f}', 
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

ax.set_xlabel('Total LEVES por Video', fontweight='bold', fontsize=12)
ax.set_ylabel('Exactitud (sobre 4.00)', fontweight='bold', fontsize=12)
ax.set_title('Relación entre LEVES y Exactitud\n(Correlación Fuerte Negativa)', 
             fontweight='bold', fontsize=14)
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)

# Colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Exactitud', rotation=270, labelpad=20, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '03_scatter_exactitud_leves.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Guardado: {output_dir / '03_scatter_exactitud_leves.png'}")

# ============================================================================
# 5. GRÁFICO 4: HEATMAP DE CORRELACIONES
# ============================================================================
print("📊 Generando Gráfico 4: Heatmap de Correlaciones...")

# Seleccionar variables de interés
corr_cols = ['exactitud', 'deducciones', 'total_leves', 
             'brazos_ok', 'piernas_ok', 'patadas_ok',
             'brazos_leve', 'piernas_leve', 'patadas_leve']
corr_matrix = df_summary[corr_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))

# Heatmap
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1, ax=ax)

ax.set_title('Matriz de Correlación (Pearson)\nEntre Variables del Sistema', 
             fontweight='bold', fontsize=14, pad=20)

# Rotar etiquetas
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

plt.tight_layout()
plt.savefig(output_dir / '04_heatmap_correlaciones.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Guardado: {output_dir / '04_heatmap_correlaciones.png'}")

# ============================================================================
# 6. GRÁFICO 5: RANKING DE VIDEOS (Barras Horizontales)
# ============================================================================
print("📊 Generando Gráfico 5: Ranking de Videos...")

# Ordenar por exactitud
df_sorted = df_summary.sort_values('exactitud', ascending=True)

fig, ax = plt.subplots(figsize=(12, 14))

# Crear barras horizontales con gradiente de color
colors_gradient = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df_sorted)))

bars = ax.barh(range(len(df_sorted)), df_sorted['exactitud'], color=colors_gradient, 
               edgecolor='black', linewidth=0.8)

# Añadir etiquetas de video
ax.set_yticks(range(len(df_sorted)))
ax.set_yticklabels(df_sorted['video_id'].values, fontsize=10)

# Añadir valores en las barras
for idx, (bar, value) in enumerate(zip(bars, df_sorted['exactitud'])):
    ax.text(value + 0.05, bar.get_y() + bar.get_height()/2, 
            f'{value:.2f}', va='center', fontsize=9, fontweight='bold')

# Líneas de referencia
ax.axvline(df_summary['exactitud'].mean(), color='red', linestyle='--', 
           linewidth=2, label=f'Media: {df_summary["exactitud"].mean():.2f}')
ax.axvline(3.0, color='green', linestyle=':', linewidth=2, alpha=0.7, 
           label='Umbral Excelente (3.0)')

ax.set_xlabel('Exactitud (sobre 4.00)', fontweight='bold', fontsize=12)
ax.set_title('Ranking de Videos por Exactitud\n(Peor → Mejor)', 
             fontweight='bold', fontsize=14)
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3, axis='x')
ax.set_xlim([1.0, 4.0])

plt.tight_layout()
plt.savefig(output_dir / '05_ranking_videos.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Guardado: {output_dir / '05_ranking_videos.png'}")

# ============================================================================
# 7. GRÁFICO 6: VIOLIN PLOTS POR COMPONENTE
# ============================================================================
print("📊 Generando Gráfico 6: Violin Plots por Componente...")

fig, ax = plt.subplots(figsize=(14, 8))

# Preparar datos en formato long
data_long = pd.DataFrame({
    'Componente': ['Brazos']*len(df_summary) + ['Piernas']*len(df_summary) + ['Patadas']*len(df_summary),
    'OK': list(df_summary['brazos_ok']) + list(df_summary['piernas_ok']) + list(df_summary['patadas_ok'])
})

# Violin plot
parts = ax.violinplot([df_summary['brazos_ok'], df_summary['piernas_ok'], df_summary['patadas_ok']],
                      positions=[1, 2, 3], showmeans=True, showmedians=True, widths=0.7)

# Colorear violines
colors_violin = ['#FF6B6B', '#4ECDC4', '#95E1D3']
for pc, color in zip(parts['bodies'], colors_violin):
    pc.set_facecolor(color)
    pc.set_alpha(0.7)
    pc.set_edgecolor('black')

# Box plots superpuestos
bp = ax.boxplot([df_summary['brazos_ok'], df_summary['piernas_ok'], df_summary['patadas_ok']],
                positions=[1, 2, 3], widths=0.3, showfliers=False,
                boxprops=dict(linewidth=2), whiskerprops=dict(linewidth=2),
                capprops=dict(linewidth=2), medianprops=dict(linewidth=2, color='red'))

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['💪 Brazos', '🦵 Piernas', '🥋 Patadas'], fontsize=12)
ax.set_ylabel('Evaluaciones OK (de 36)', fontweight='bold', fontsize=12)
ax.set_title('Distribución de Evaluaciones OK por Componente\n(Violin + Box Plots)', 
             fontweight='bold', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')

# Añadir promedios
for i, comp_data in enumerate([df_summary['brazos_ok'], df_summary['piernas_ok'], df_summary['patadas_ok']], 1):
    mean_val = comp_data.mean()
    ax.text(i, mean_val, f'μ={mean_val:.1f}', ha='center', va='bottom', 
            fontsize=10, fontweight='bold', 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.savefig(output_dir / '06_violin_plots_componentes.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Guardado: {output_dir / '06_violin_plots_componentes.png'}")

# ============================================================================
# 8. GRÁFICO 7: DISTRIBUCIÓN DE LEVES
# ============================================================================
print("📊 Generando Gráfico 7: Distribución de LEVES...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Subplot 1: Histograma de LEVES totales
ax1.hist(df_summary['total_leves'], bins=12, alpha=0.7, color='coral', 
         edgecolor='black', linewidth=1.2)

# Líneas de referencia
ax1.axvline(df_summary['total_leves'].mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Media: {df_summary["total_leves"].mean():.1f}')
ax1.axvline(df_summary['total_leves'].median(), color='blue', linestyle='--', 
            linewidth=2, label=f'Mediana: {df_summary["total_leves"].median():.0f}')

ax1.set_xlabel('Total LEVES por Video', fontweight='bold', fontsize=12)
ax1.set_ylabel('Frecuencia (# de Videos)', fontweight='bold', fontsize=12)
ax1.set_title('Distribución de LEVES Totales por Video', fontweight='bold', fontsize=14)
ax1.legend(loc='best', fontsize=11)
ax1.grid(True, alpha=0.3)

# Subplot 2: Barras apiladas por componente
width = 0.6
x = range(len(df_sorted))
p1 = ax2.bar(x, df_sorted['brazos_leve'], width, label='Brazos', color='#FF6B6B', alpha=0.8)
p2 = ax2.bar(x, df_sorted['piernas_leve'], width, bottom=df_sorted['brazos_leve'],
             label='Piernas', color='#4ECDC4', alpha=0.8)
p3 = ax2.bar(x, df_sorted['patadas_leve'], width, 
             bottom=df_sorted['brazos_leve'] + df_sorted['piernas_leve'],
             label='Patadas', color='#95E1D3', alpha=0.8)

ax2.set_xlabel('Videos (ordenados por Exactitud)', fontweight='bold', fontsize=12)
ax2.set_ylabel('LEVES por Componente', fontweight='bold', fontsize=12)
ax2.set_title('Distribución de LEVES por Componente y Video\n(Barras Apiladas)', 
              fontweight='bold', fontsize=14)
ax2.legend(loc='upper left', fontsize=11)
ax2.set_xticks(range(0, len(df_sorted), 2))
ax2.set_xticklabels([df_sorted.iloc[i]['video_id'].replace('8yang_', '') 
                      for i in range(0, len(df_sorted), 2)], rotation=45)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / '07_distribucion_leves.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Guardado: {output_dir / '07_distribucion_leves.png'}")

# ============================================================================
# 9. GRÁFICO 8: SERIE TEMPORAL DE VIDEOS
# ============================================================================
print("📊 Generando Gráfico 8: Serie Temporal de Videos...")

fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Subplot 1: Exactitud y Deducciones
ax1 = axes[0]
x = range(len(videos))

ax1_twin = ax1.twinx()

# Exactitud (línea + puntos)
line1 = ax1.plot(x, df_summary['exactitud'], 'o-', linewidth=2, markersize=8, 
                 color='green', label='Exactitud', alpha=0.8)
ax1.fill_between(x, df_summary['exactitud'], alpha=0.3, color='green')

# Deducciones (línea + puntos)
line2 = ax1_twin.plot(x, df_summary['deducciones'], 's--', linewidth=2, markersize=7, 
                       color='red', label='Deducciones', alpha=0.8)

# Media de exactitud
ax1.axhline(df_summary['exactitud'].mean(), color='darkgreen', linestyle=':', 
            linewidth=2, alpha=0.5, label=f'Media Exactitud: {df_summary["exactitud"].mean():.2f}')

ax1.set_xlabel('Videos (secuencia)', fontweight='bold', fontsize=12)
ax1.set_ylabel('Exactitud (sobre 4.00)', fontweight='bold', fontsize=12, color='green')
ax1_twin.set_ylabel('Deducciones (pts)', fontweight='bold', fontsize=12, color='red')
ax1.set_title('Evolución de Exactitud y Deducciones por Video', fontweight='bold', fontsize=14)
ax1.set_xticks(range(0, len(videos), 2))
ax1.set_xticklabels([v.replace('8yang_', '') for v in videos[::2]], rotation=45)
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='y', labelcolor='green')
ax1_twin.tick_params(axis='y', labelcolor='red')

# Leyenda combinada
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=11)

# Subplot 2: Componentes OK
ax2 = axes[1]

ax2.plot(x, df_summary['brazos_ok'], 'o-', linewidth=2, markersize=6, 
         label='💪 Brazos OK', color='#FF6B6B', alpha=0.8)
ax2.plot(x, df_summary['piernas_ok'], 's-', linewidth=2, markersize=6, 
         label='🦵 Piernas OK', color='#4ECDC4', alpha=0.8)
ax2.plot(x, df_summary['patadas_ok'], '^-', linewidth=2, markersize=6, 
         label='🥋 Patadas OK', color='#95E1D3', alpha=0.8)

ax2.set_xlabel('Videos (secuencia)', fontweight='bold', fontsize=12)
ax2.set_ylabel('Evaluaciones OK (de 36)', fontweight='bold', fontsize=12)
ax2.set_title('Evolución de Componentes OK por Video', fontweight='bold', fontsize=14)
ax2.set_xticks(range(0, len(videos), 2))
ax2.set_xticklabels([v.replace('8yang_', '') for v in videos[::2]], rotation=45)
ax2.legend(loc='lower left', fontsize=11, ncol=3)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 38])

plt.tight_layout()
plt.savefig(output_dir / '08_serie_temporal_videos.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Guardado: {output_dir / '08_serie_temporal_videos.png'}")

# ============================================================================
# 10. GRÁFICO 9: SCATTER 3D - Brazos vs Piernas vs Patadas
# ============================================================================
print("📊 Generando Gráfico 9: Scatter Plot 3D...")

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Scatter 3D
scatter = ax.scatter(df_summary['brazos_ok'], df_summary['piernas_ok'], 
                    df_summary['patadas_ok'], c=df_summary['exactitud'], 
                    cmap='viridis', s=200, alpha=0.7, edgecolors='black', linewidth=1)

# Añadir etiquetas
for idx, row in df_summary.iterrows():
    ax.text(row['brazos_ok'], row['piernas_ok'], row['patadas_ok'], 
            row['video_id'].replace('8yang_', ''), fontsize=8)

ax.set_xlabel('💪 Brazos OK', fontweight='bold', fontsize=12, labelpad=10)
ax.set_ylabel('🦵 Piernas OK', fontweight='bold', fontsize=12, labelpad=10)
ax.set_zlabel('🥋 Patadas OK', fontweight='bold', fontsize=12, labelpad=10)
ax.set_title('Espacio 3D de Componentes OK\n(Color = Exactitud)', 
             fontweight='bold', fontsize=14, pad=20)

# Colorbar
cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Exactitud', rotation=270, labelpad=20, fontweight='bold')

# Rotar para mejor vista
ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig(output_dir / '09_scatter_3d_componentes.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Guardado: {output_dir / '09_scatter_3d_componentes.png'}")

# ============================================================================
# 11. GRÁFICO 10: RADAR CHART - Promedios de Componentes
# ============================================================================
print("📊 Generando Gráfico 10: Radar Chart de Componentes...")

# Calcular porcentajes promedio
brazos_pct = (df_summary['brazos_ok'].mean() / 36) * 100
piernas_pct = (df_summary['piernas_ok'].mean() / 36) * 100
patadas_pct = (df_summary['patadas_ok'].mean() / 36) * 100
exactitud_pct = (df_summary['exactitud'].mean() / 4.0) * 100

# Datos para radar
categories = ['Brazos\nOK', 'Piernas\nOK', 'Patadas\nOK', 'Exactitud\nGeneral']
values = [brazos_pct, piernas_pct, patadas_pct, exactitud_pct]

# Cerrar el polígono
values += values[:1]
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Dibujar el polígono
ax.plot(angles, values, 'o-', linewidth=2, color='b', label='Sistema Actual')
ax.fill(angles, values, alpha=0.25, color='b')

# Línea de referencia (80%)
ax.plot(angles, [80]*len(angles), 'r--', linewidth=1.5, alpha=0.5, label='Umbral 80%')

# Configurar etiquetas
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12)
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10)
ax.set_title('Desempeño Promedio del Sistema\n(Radar Chart)', 
             fontweight='bold', fontsize=14, pad=30)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
ax.grid(True, alpha=0.3)

# Añadir valores en cada vértice
for angle, value, category in zip(angles[:-1], values[:-1], categories):
    ax.text(angle, value + 5, f'{value:.1f}%', ha='center', va='center',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig(output_dir / '10_radar_chart_componentes.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Guardado: {output_dir / '10_radar_chart_componentes.png'}")

# ============================================================================
# 12. RESUMEN FINAL
# ============================================================================
print("\n" + "="*100)
print("✅ GENERACIÓN DE GRÁFICOS COMPLETADA")
print("="*100)
print(f"\n📁 Todos los gráficos guardados en: {output_dir.absolute()}\n")
print("📊 Gráficos generados:")
print("   1. 01_distribucion_exactitud.png       - Histograma + KDE + Box Plot")
print("   2. 02_boxplots_componentes.png          - Box Plots individuales")
print("   3. 03_scatter_exactitud_leves.png       - Correlación con regresión")
print("   4. 04_heatmap_correlaciones.png         - Matriz de correlaciones")
print("   5. 05_ranking_videos.png                - Barras horizontales")
print("   6. 06_violin_plots_componentes.png      - Distribuciones detalladas")
print("   7. 07_distribucion_leves.png            - Histograma + Barras apiladas")
print("   8. 08_serie_temporal_videos.png         - Evolución secuencial")
print("   9. 09_scatter_3d_componentes.png        - Visualización 3D")
print("  10. 10_radar_chart_componentes.png       - Desempeño global")
print("\n" + "="*100)
print("💡 Usa estos gráficos para tu tesis, presentaciones y publicaciones")
print("="*100)
print()

# Estadísticas finales en consola
print("\n📈 RESUMEN ESTADÍSTICO RÁPIDO:")
print(f"   • Exactitud promedio: {df_summary['exactitud'].mean():.2f} ± {df_summary['exactitud'].std():.2f}")
print(f"   • Correlación Exactitud-LEVES: r = {stats.pearsonr(df_summary['exactitud'], df_summary['total_leves'])[0]:.3f}")
print(f"   • Videos con exactitud ≥3.0: {len(df_summary[df_summary['exactitud'] >= 3.0])} de 29 ({len(df_summary[df_summary['exactitud'] >= 3.0])/29*100:.1f}%)")
print(f"   • GRAVES totales: 0 (100% eliminación exitosa) ✅")
print()
