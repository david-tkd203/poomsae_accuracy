#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparación Entre Dos Evaluadores del Mismo Video
==================================================

Compara los reportes de dos personas que evaluaron el mismo video de Poomsae.

Análisis generado:
- Concordancia general (Cohen's Kappa)
- Porcentaje de acuerdo por componente
- Identificación de diferencias específicas
- Matriz de confusión por componente
- Visualizaciones comparativas
- Reporte detallado de discrepancias

Uso:
    python comparar_evaluadores.py <video_id> <reporte1.xlsx> <reporte2.xlsx>
    
Ejemplo:
    python comparar_evaluadores.py 8yang_001 reports/8yang_001_score.xlsx reports/8yang_001_evaluador2.xlsx
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11


def leer_reporte(path):
    """Lee un reporte Excel y retorna el dataframe de detalle"""
    try:
        df = pd.read_excel(path, sheet_name="detalle")
        df_resumen = pd.read_excel(path, sheet_name="resumen")
        return df, df_resumen
    except Exception as e:
        print(f"❌ Error leyendo {path}: {e}")
        sys.exit(1)


def calcular_kappa(eval1, eval2):
    """Calcula Cohen's Kappa entre dos evaluadores"""
    return cohen_kappa_score(eval1, eval2)


def calcular_porcentaje_acuerdo(eval1, eval2):
    """Calcula el porcentaje de acuerdo simple"""
    return (eval1 == eval2).sum() / len(eval1) * 100


def mapear_evaluacion_a_numero(eval_series):
    """Mapea OK/LEVE/GRAVE a valores numéricos"""
    mapping = {'OK': 0, 'LEVE': 1, 'GRAVE': 2}
    return eval_series.map(mapping)


def generar_matriz_confusion(eval1, eval2, componente_nombre, output_dir):
    """Genera y guarda matriz de confusión"""
    labels = ['OK', 'LEVE', 'GRAVE']
    cm = confusion_matrix(eval1, eval2, labels=labels)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, 
                yticklabels=labels, ax=ax, cbar_kws={'label': 'Cantidad'})
    
    ax.set_xlabel('Evaluador 2', fontweight='bold', fontsize=12)
    ax.set_ylabel('Evaluador 1', fontweight='bold', fontsize=12)
    ax.set_title(f'Matriz de Confusión: {componente_nombre}\n(Diagonal = Acuerdos)', 
                 fontweight='bold', fontsize=13)
    
    # Añadir porcentaje de acuerdo
    total = cm.sum()
    acuerdo = cm.diagonal().sum()
    pct_acuerdo = (acuerdo / total) * 100
    
    ax.text(0.5, -0.15, f'Acuerdo: {acuerdo}/{total} ({pct_acuerdo:.1f}%)', 
            transform=ax.transAxes, ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    filename = f'confusion_{componente_nombre.lower().replace(" ", "_")}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm, pct_acuerdo


def identificar_diferencias(df1, df2, componente):
    """Identifica movimientos donde hubo diferencias"""
    diferencias = []
    for idx in range(len(df1)):
        eval1 = df1.iloc[idx][componente]
        eval2 = df2.iloc[idx][componente]
        
        if eval1 != eval2:
            diferencias.append({
                'movimiento': df1.iloc[idx]['M'] if 'M' in df1.columns else idx,
                'tecnica': df1.iloc[idx]['tech_es'] if 'tech_es' in df1.columns else 'N/A',
                'evaluador1': eval1,
                'evaluador2': eval2,
                'gravedad_diferencia': abs(mapear_evaluacion_a_numero(pd.Series([eval1])).iloc[0] - 
                                          mapear_evaluacion_a_numero(pd.Series([eval2])).iloc[0])
            })
    
    return pd.DataFrame(diferencias)


def generar_grafico_comparativo_barras(stats_comp, output_dir):
    """Genera gráfico de barras comparando porcentajes de OK/LEVE/GRAVE"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    componentes = ['Brazos', 'Piernas', 'Patadas']
    keys = ['comp_arms', 'comp_legs', 'comp_kick']
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    emojis = ['💪', '🦵', '🥋']
    
    for idx, (comp_name, key, color, emoji) in enumerate(zip(componentes, keys, colors, emojis)):
        ax = axes[idx]
        
        stats = stats_comp[key]
        
        x = np.arange(3)
        width = 0.35
        
        eval1_vals = [stats['eval1']['OK'], stats['eval1']['LEVE'], stats['eval1']['GRAVE']]
        eval2_vals = [stats['eval2']['OK'], stats['eval2']['LEVE'], stats['eval2']['GRAVE']]
        
        bars1 = ax.bar(x - width/2, eval1_vals, width, label='Evaluador 1', 
                      color=color, alpha=0.8, edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, eval2_vals, width, label='Evaluador 2', 
                      color=color, alpha=0.5, edgecolor='black', linewidth=1)
        
        ax.set_ylabel('Cantidad de Evaluaciones', fontweight='bold')
        ax.set_title(f'{emoji} {comp_name}', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(['OK', 'LEVE', 'GRAVE'])
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Añadir valores en las barras
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('Comparación de Distribución de Evaluaciones\nEntre Evaluadores', 
                 fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'comparacion_barras_componentes.png', dpi=300, bbox_inches='tight')
    plt.close()


def generar_grafico_concordancia(kappas, acuerdos, output_dir):
    """Genera gráfico de concordancia (Kappa y % acuerdo)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    componentes = ['Brazos', 'Piernas', 'Patadas', 'GENERAL']
    x = np.arange(len(componentes))
    
    # Subplot 1: Cohen's Kappa
    colors_kappa = ['#2ecc71' if k >= 0.8 else '#f39c12' if k >= 0.6 else '#e74c3c' 
                    for k in kappas]
    
    bars1 = ax1.bar(x, kappas, color=colors_kappa, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Líneas de referencia para interpretación de Kappa
    ax1.axhline(0.8, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Excelente (≥0.8)')
    ax1.axhline(0.6, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Bueno (≥0.6)')
    ax1.axhline(0.4, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Moderado (≥0.4)')
    
    ax1.set_ylabel("Cohen's Kappa", fontweight='bold', fontsize=12)
    ax1.set_title("Concordancia (Cohen's Kappa)\nEntre Evaluadores", fontweight='bold', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(componentes, fontsize=11)
    ax1.set_ylim([0, 1])
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores
    for bar, kappa in zip(bars1, kappas):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{kappa:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Subplot 2: % de Acuerdo
    colors_acuerdo = ['#2ecc71' if a >= 90 else '#f39c12' if a >= 80 else '#e74c3c' 
                      for a in acuerdos]
    
    bars2 = ax2.bar(x, acuerdos, color=colors_acuerdo, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Líneas de referencia
    ax2.axhline(90, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Excelente (≥90%)')
    ax2.axhline(80, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Bueno (≥80%)')
    
    ax2.set_ylabel('Porcentaje de Acuerdo (%)', fontweight='bold', fontsize=12)
    ax2.set_title('Acuerdo Simple (%)\nEntre Evaluadores', fontweight='bold', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(componentes, fontsize=11)
    ax2.set_ylim([0, 100])
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores
    for bar, acuerdo in zip(bars2, acuerdos):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acuerdo:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'concordancia_evaluadores.png', dpi=300, bbox_inches='tight')
    plt.close()


def generar_grafico_diferencias(df_dif_brazos, df_dif_piernas, df_dif_patadas, output_dir):
    """Genera gráfico de diferencias por movimiento"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    componentes_data = [
        (df_dif_brazos, '💪 Brazos', '#FF6B6B'),
        (df_dif_piernas, '🦵 Piernas', '#4ECDC4'),
        (df_dif_patadas, '🥋 Patadas', '#95E1D3')
    ]
    
    for idx, (df_dif, titulo, color) in enumerate(componentes_data):
        ax = axes[idx]
        
        if len(df_dif) == 0:
            ax.text(0.5, 0.5, '✅ Sin diferencias', ha='center', va='center',
                   fontsize=16, fontweight='bold', transform=ax.transAxes)
            ax.set_title(titulo, fontweight='bold', fontsize=13)
            ax.axis('off')
            continue
        
        # Crear gráfico de barras con las diferencias
        x_pos = range(len(df_dif))
        bars = ax.bar(x_pos, [1] * len(df_dif), color=color, alpha=0.7, 
                     edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Índice de Diferencia', fontweight='bold', fontsize=11)
        ax.set_ylabel('Diferencia Detectada', fontweight='bold', fontsize=11)
        ax.set_title(f'{titulo} - Diferencias Encontradas', fontweight='bold', fontsize=13)
        ax.set_ylim([0, 1.5])
        ax.set_yticks([])
        ax.grid(True, alpha=0.3, axis='x')
        
        # Añadir etiquetas de movimiento
        for i, (bar, mov) in enumerate(zip(bars, df_dif['movimiento'])):
            ax.text(bar.get_x() + bar.get_width()/2., 0.5,
                   f'M{mov}',
                   ha='center', va='center', fontsize=8, fontweight='bold', rotation=0)
        
        # Añadir estadísticas
        total_dif = len(df_dif)
        gravedad_alta = len(df_dif[df_dif['gravedad_diferencia'] == 2])
        ax.text(0.02, 0.98, f'Total diferencias: {total_dif}\nDiferencias graves: {gravedad_alta}', 
                transform=ax.transAxes, va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'diferencias_por_movimiento.png', dpi=300, bbox_inches='tight')
    plt.close()


def generar_reporte_excel(video_id, df1, df2, df_resumen1, df_resumen2, 
                          stats_comp, kappas, acuerdos, 
                          df_dif_brazos, df_dif_piernas, df_dif_patadas, 
                          output_dir):
    """Genera reporte Excel con comparación detallada"""
    output_file = output_dir / f'{video_id}_comparacion_evaluadores.xlsx'
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Hoja 1: Resumen de concordancia
        resumen_data = {
            'Componente': ['Brazos', 'Piernas', 'Patadas', 'GENERAL'],
            'Cohen_Kappa': kappas,
            'Porcentaje_Acuerdo': acuerdos,
            'Interpretación_Kappa': [
                'Excelente' if k >= 0.8 else 'Bueno' if k >= 0.6 else 'Moderado' if k >= 0.4 else 'Pobre'
                for k in kappas
            ]
        }
        df_resumen_conc = pd.DataFrame(resumen_data)
        df_resumen_conc.to_excel(writer, sheet_name='resumen_concordancia', index=False)
        
        # Hoja 2: Resumen de cada evaluador
        resumen_eval_data = {
            'Métrica': ['Exactitud Final', 'Deducciones Totales', 'Brazos OK', 'Piernas OK', 'Patadas OK',
                       'Brazos LEVE', 'Piernas LEVE', 'Patadas LEVE',
                       'Brazos GRAVE', 'Piernas GRAVE', 'Patadas GRAVE'],
            'Evaluador_1': [
                df_resumen1['exactitud_final'].iloc[0],
                df_resumen1['ded_total'].iloc[0],
                stats_comp['comp_arms']['eval1']['OK'],
                stats_comp['comp_legs']['eval1']['OK'],
                stats_comp['comp_kick']['eval1']['OK'],
                stats_comp['comp_arms']['eval1']['LEVE'],
                stats_comp['comp_legs']['eval1']['LEVE'],
                stats_comp['comp_kick']['eval1']['LEVE'],
                stats_comp['comp_arms']['eval1']['GRAVE'],
                stats_comp['comp_legs']['eval1']['GRAVE'],
                stats_comp['comp_kick']['eval1']['GRAVE']
            ],
            'Evaluador_2': [
                df_resumen2['exactitud_final'].iloc[0],
                df_resumen2['ded_total'].iloc[0],
                stats_comp['comp_arms']['eval2']['OK'],
                stats_comp['comp_legs']['eval2']['OK'],
                stats_comp['comp_kick']['eval2']['OK'],
                stats_comp['comp_arms']['eval2']['LEVE'],
                stats_comp['comp_legs']['eval2']['LEVE'],
                stats_comp['comp_kick']['eval2']['LEVE'],
                stats_comp['comp_arms']['eval2']['GRAVE'],
                stats_comp['comp_legs']['eval2']['GRAVE'],
                stats_comp['comp_kick']['eval2']['GRAVE']
            ]
        }
        df_resumen_eval = pd.DataFrame(resumen_eval_data)
        df_resumen_eval['Diferencia'] = df_resumen_eval['Evaluador_2'] - df_resumen_eval['Evaluador_1']
        df_resumen_eval.to_excel(writer, sheet_name='resumen_evaluadores', index=False)
        
        # Hoja 3-5: Diferencias por componente
        if len(df_dif_brazos) > 0:
            df_dif_brazos.to_excel(writer, sheet_name='diferencias_brazos', index=False)
        if len(df_dif_piernas) > 0:
            df_dif_piernas.to_excel(writer, sheet_name='diferencias_piernas', index=False)
        if len(df_dif_patadas) > 0:
            df_dif_patadas.to_excel(writer, sheet_name='diferencias_patadas', index=False)
        
        # Hoja 6: Comparación movimiento por movimiento
        df_comp = pd.DataFrame({
            'movimiento': df1['M'] if 'M' in df1.columns else range(len(df1)),
            'tecnica': df1['tech_es'] if 'tech_es' in df1.columns else 'N/A',
            'brazos_eval1': df1['comp_arms'],
            'brazos_eval2': df2['comp_arms'],
            'brazos_coincide': df1['comp_arms'] == df2['comp_arms'],
            'piernas_eval1': df1['comp_legs'],
            'piernas_eval2': df2['comp_legs'],
            'piernas_coincide': df1['comp_legs'] == df2['comp_legs'],
            'patadas_eval1': df1['comp_kick'],
            'patadas_eval2': df2['comp_kick'],
            'patadas_coincide': df1['comp_kick'] == df2['comp_kick']
        })
        df_comp.to_excel(writer, sheet_name='comparacion_detallada', index=False)
    
    print(f"   ✅ Reporte Excel guardado: {output_file}")
    return output_file


def main():
    print("="*100)
    print(" 👥 COMPARACIÓN ENTRE DOS EVALUADORES DEL MISMO VIDEO")
    print("="*100)
    print()
    
    # Verificar argumentos
    if len(sys.argv) != 4:
        print("❌ Uso incorrecto")
        print()
        print("Uso:")
        print("   python comparar_evaluadores.py <video_id> <reporte1.xlsx> <reporte2.xlsx>")
        print()
        print("Ejemplo:")
        print("   python comparar_evaluadores.py 8yang_001 reports/8yang_001_score.xlsx reports/8yang_001_evaluador2.xlsx")
        print()
        sys.exit(1)
    
    video_id = sys.argv[1]
    reporte1_path = Path(sys.argv[2])
    reporte2_path = Path(sys.argv[3])
    
    # Verificar existencia de archivos
    if not reporte1_path.exists():
        print(f"❌ Error: No se encontró {reporte1_path}")
        sys.exit(1)
    
    if not reporte2_path.exists():
        print(f"❌ Error: No se encontró {reporte2_path}")
        sys.exit(1)
    
    print(f"📹 Video: {video_id}")
    print(f"📄 Evaluador 1: {reporte1_path}")
    print(f"📄 Evaluador 2: {reporte2_path}")
    print()
    
    # Crear directorio de salida
    output_dir = Path(f"comparacion_{video_id}")
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Directorio de salida: {output_dir.absolute()}")
    print()
    
    # Leer reportes
    print("📖 Leyendo reportes...")
    df1, df_resumen1 = leer_reporte(reporte1_path)
    df2, df_resumen2 = leer_reporte(reporte2_path)
    
    # Verificar que tienen el mismo número de movimientos
    if len(df1) != len(df2):
        print(f"⚠️  ADVERTENCIA: Los reportes tienen diferente número de movimientos")
        print(f"   Evaluador 1: {len(df1)} movimientos")
        print(f"   Evaluador 2: {len(df2)} movimientos")
        print(f"   Se compararán solo los primeros {min(len(df1), len(df2))} movimientos")
        min_len = min(len(df1), len(df2))
        df1 = df1.iloc[:min_len]
        df2 = df2.iloc[:min_len]
    
    print(f"✅ {len(df1)} movimientos a comparar")
    print()
    
    # Estadísticas por componente
    print("📊 Calculando estadísticas por componente...")
    componentes = ['comp_arms', 'comp_legs', 'comp_kick']
    stats_comp = {}
    
    for comp in componentes:
        stats_comp[comp] = {
            'eval1': {
                'OK': (df1[comp] == 'OK').sum(),
                'LEVE': (df1[comp] == 'LEVE').sum(),
                'GRAVE': (df1[comp] == 'GRAVE').sum()
            },
            'eval2': {
                'OK': (df2[comp] == 'OK').sum(),
                'LEVE': (df2[comp] == 'LEVE').sum(),
                'GRAVE': (df2[comp] == 'GRAVE').sum()
            }
        }
    print()
    
    # Calcular métricas de concordancia
    print("🔍 Calculando métricas de concordancia...")
    
    kappas = []
    acuerdos = []
    
    for comp, nombre in zip(componentes, ['Brazos', 'Piernas', 'Patadas']):
        kappa = calcular_kappa(df1[comp], df2[comp])
        acuerdo = calcular_porcentaje_acuerdo(df1[comp], df2[comp])
        
        kappas.append(kappa)
        acuerdos.append(acuerdo)
        
        print(f"   {nombre}:")
        print(f"      - Cohen's Kappa: {kappa:.3f}")
        print(f"      - % Acuerdo: {acuerdo:.1f}%")
    
    # Concordancia general (todas las evaluaciones)
    all_eval1 = pd.concat([df1[c] for c in componentes])
    all_eval2 = pd.concat([df2[c] for c in componentes])
    kappa_general = calcular_kappa(all_eval1, all_eval2)
    acuerdo_general = calcular_porcentaje_acuerdo(all_eval1, all_eval2)
    
    kappas.append(kappa_general)
    acuerdos.append(acuerdo_general)
    
    print(f"   GENERAL:")
    print(f"      - Cohen's Kappa: {kappa_general:.3f}")
    print(f"      - % Acuerdo: {acuerdo_general:.1f}%")
    print()
    
    # Identificar diferencias
    print("🔎 Identificando diferencias específicas...")
    df_dif_brazos = identificar_diferencias(df1, df2, 'comp_arms')
    df_dif_piernas = identificar_diferencias(df1, df2, 'comp_legs')
    df_dif_patadas = identificar_diferencias(df1, df2, 'comp_kick')
    
    print(f"   Brazos: {len(df_dif_brazos)} diferencias")
    print(f"   Piernas: {len(df_dif_piernas)} diferencias")
    print(f"   Patadas: {len(df_dif_patadas)} diferencias")
    print(f"   TOTAL: {len(df_dif_brazos) + len(df_dif_piernas) + len(df_dif_patadas)} diferencias")
    print()
    
    # Generar visualizaciones
    print("📊 Generando visualizaciones...")
    
    # 1. Matrices de confusión
    for comp, nombre in zip(componentes, ['Brazos', 'Piernas', 'Patadas']):
        generar_matriz_confusion(df1[comp], df2[comp], nombre, output_dir)
    print("   ✅ Matrices de confusión generadas")
    
    # 2. Gráfico de barras comparativo
    generar_grafico_comparativo_barras(stats_comp, output_dir)
    print("   ✅ Gráfico de barras comparativas generado")
    
    # 3. Gráfico de concordancia
    generar_grafico_concordancia(kappas, acuerdos, output_dir)
    print("   ✅ Gráfico de concordancia generado")
    
    # 4. Gráfico de diferencias
    generar_grafico_diferencias(df_dif_brazos, df_dif_piernas, df_dif_patadas, output_dir)
    print("   ✅ Gráfico de diferencias generado")
    print()
    
    # Generar reporte Excel
    print("📝 Generando reporte Excel...")
    excel_file = generar_reporte_excel(video_id, df1, df2, df_resumen1, df_resumen2,
                                       stats_comp, kappas, acuerdos,
                                       df_dif_brazos, df_dif_piernas, df_dif_patadas,
                                       output_dir)
    print()
    
    # Resumen final
    print("="*100)
    print(" ✅ COMPARACIÓN COMPLETADA")
    print("="*100)
    print()
    print(f"📁 Resultados guardados en: {output_dir.absolute()}")
    print()
    print("📊 Archivos generados:")
    print("   • confusion_brazos.png")
    print("   • confusion_piernas.png")
    print("   • confusion_patadas.png")
    print("   • comparacion_barras_componentes.png")
    print("   • concordancia_evaluadores.png")
    print("   • diferencias_por_segmento.png")
    print(f"   • {video_id}_comparacion_evaluadores.xlsx")
    print()
    print("📈 RESUMEN DE CONCORDANCIA:")
    print(f"   • Cohen's Kappa General: {kappa_general:.3f} ", end="")
    if kappa_general >= 0.8:
        print("(⭐ EXCELENTE)")
    elif kappa_general >= 0.6:
        print("(✅ BUENO)")
    elif kappa_general >= 0.4:
        print("(⚠️  MODERADO)")
    else:
        print("(❌ POBRE)")
    
    print(f"   • Acuerdo Simple: {acuerdo_general:.1f}%")
    print(f"   • Diferencias Totales: {len(df_dif_brazos) + len(df_dif_piernas) + len(df_dif_patadas)}/{len(df1)*3} evaluaciones")
    print()
    print("💡 Interpreta el Kappa:")
    print("   • 0.81-1.00: Acuerdo casi perfecto")
    print("   • 0.61-0.80: Acuerdo sustancial")
    print("   • 0.41-0.60: Acuerdo moderado")
    print("   • 0.21-0.40: Acuerdo débil")
    print("   • 0.00-0.20: Acuerdo ligero")
    print("="*100)


if __name__ == "__main__":
    main()
