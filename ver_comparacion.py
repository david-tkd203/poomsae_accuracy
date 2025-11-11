import pandas as pd

xl = pd.ExcelFile('comparacion_8yang_010/8yang_010_comparacion_evaluadores.xlsx')

print('='*80)
print(' 📋 CONTENIDO DEL REPORTE DE COMPARACIÓN')
print('='*80)
print()

print('📄 Hojas disponibles:')
for sheet in xl.sheet_names:
    print(f'   • {sheet}')
print()

print('='*80)
print(' 📊 RESUMEN DE CONCORDANCIA')
print('='*80)
df = pd.read_excel(xl, 'resumen_concordancia')
print(df.to_string(index=False))
print()

print('='*80)
print(' 📊 RESUMEN POR EVALUADOR')
print('='*80)
df2 = pd.read_excel(xl, 'resumen_evaluadores')
print(df2.to_string(index=False))
print()

print('='*80)
print(' 🔍 DIFERENCIAS EN PIERNAS (primeras 5)')
print('='*80)
if 'diferencias_piernas' in xl.sheet_names:
    df3 = pd.read_excel(xl, 'diferencias_piernas')
    print(df3.to_string(index=False))
else:
    print('Sin diferencias en piernas')
print()

print('='*80)
print(' 🔍 DIFERENCIAS EN PATADAS (primeras 5)')
print('='*80)
if 'diferencias_patadas' in xl.sheet_names:
    df4 = pd.read_excel(xl, 'diferencias_patadas')
    print(df4.to_string(index=False))
else:
    print('Sin diferencias en patadas')
