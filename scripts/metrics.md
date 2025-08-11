# 🚦 **CHEAT SHEET: Sistema de Análisis de Pairs para Memecoins**

*Guía Rápida de Interpretación de Métricas*

---

## **🎯 SISTEMA DE SEMÁFOROS**

| Color | Significado | Confianza | Acción |
|-------|-------------|-----------|--------|
| **🟦 AZUL** | **ÓPTIMO** | 95%+ | Trade posición completa |
| **🟩 VERDE** | **BUENO** | 80-95% | Trade con confianza |
| **🟨 AMARILLO** | **ACEPTABLE** | 60-80% | Trade con precaución |
| **🟥 ROJO** | **EVITAR** | <60% | No tradear |

---

## **📊 THRESHOLDS POR MÉTRICA**

### **🔗 COINTEGRACIÓN (Relación Largo Plazo)**

#### **Johansen Trace Ratio**
```
🟦 ≥ 1.0    "Relación muy fuerte - Ideal long-term"
🟩 0.7-1.0  "Relación sólida - Excelente trading"  
🟨 0.4-0.7  "Relación moderada - OK memecoins"
🟥 < 0.4    "Relación débil - Solo short-term"
```

**💡 ¿Qué es?** Mide si dos activos mantienen relación estable a largo plazo  
**💡 Analogía:** ¿Qué tan unidos están dos amigos que a veces se separan pero siempre vuelven?

---

### **📊 ESTACIONARIEDAD (Spread Estable)**

#### **ADF Test (p-value)**
```
🟦 < 0.01   "Spread súper estable - Como resorte perfecto"
🟩 < 0.05   "Spread estable - Vuelve a media consistentemente"
🟨 < 0.10   "Spread algo estable - Aceptable memecoins"
🟥 ≥ 0.10   "Spread inestable - Evitar"
```

#### **KPSS Test (p-value) - ULTRA RELAJADO**
```
🟦 > 0.05   "Muy estable alrededor de media"
🟩 > 0.02   "Estable - Normal para memecoins"
🟨 > 0.005  "Aceptable - Típico alta volatilidad"
🟥 ≤ 0.005  "Inestable - Demasiado volátil"
```

**💡 ¿Qué es?** Mide si el spread (diferencia precios) es estable en el tiempo  
**💡 Analogía:** ¿Se comporta como termostato (estable) o como montaña rusa (caótico)?

---

### **🔄 MEAN REVERSION (Velocidad Corrección)**

#### **Half-Life (% del Window)**
```
🟦 6-14%    "Corrección óptima - Ni muy rápido ni lento"
🟩 ≤ 28%    "Corrección buena - Velocidad adecuada"
🟨 ≤ 45%    "Corrección lenta pero aceptable"
🟥 > 45%    "Corrección muy lenta - Riesgo alto"
```

**💡 ¿Qué es?** Tiempo promedio para que spread divergente se reduzca a la mitad  
**💡 Analogía:** Si tomas medicina, ¿cuánto tarda el efecto en reducirse a la mitad?

#### **Hurst Exponent**
```
🟦 0.32-0.48  "Mean-reverting perfecto - Como péndulo"
🟩 ≤ 0.58     "Mean-reverting bueno - Vuelve a media"
🟨 ≤ 0.68     "Débil mean-reversion - Cuidado"
🟥 > 0.68     "Trending - Malo para pair trading"
```

**💡 ¿Qué es?** Tipo de comportamiento del spread  
**💡 Analogía:** ¿Se comporta como péndulo (vuelve al centro) o como escalador (sigue subiendo)?

---

### **📈 RELACIÓN LINEAL**

#### **R-Squared (% Varianza Explicada)**
```
🟦 > 75%    "Relación muy fuerte - Predicción excelente"
🟩 > 60%    "Relación fuerte - Predicción buena"
🟨 > 40%    "Relación moderada - Suficiente memecoins"
🟥 ≤ 40%    "Relación débil - Predicción pobre"
```

**💡 ¿Qué es?** % del movimiento de Asset1 que explica Asset2  
**💡 Analogía:** Al ver las nubes, ¿qué % de la lluvia puedes predecir?

#### **Beta Stability (Desviación Estándar)**
```
🟦 < 0.05   "Relación súper estable - Proporción constante"
🟩 < 0.20   "Relación estable - Proporción predecible"
🟨 < 0.45   "Relación moderada - Aceptable memecoins"
🟥 ≥ 0.45   "Relación inestable - Proporción impredecible"
```

**💡 ¿Qué es?** Estabilidad de la proporción entre activos  
**💡 Analogía:** Al cocinar, ¿necesitas siempre la misma proporción de ingredientes?

---

### **🤝 CORRELACIÓN (Movimiento Conjunto)**

#### **Pearson Correlation**
```
🟦 > 0.60   "Se mueven muy sincronizados - Como gemelos"
🟩 > 0.45   "Se mueven bien sincronizados - Como hermanos"
🟨 > 0.25   "Sincronización débil - Como primos lejanos"
🟥 ≤ 0.25   "Sin sincronización - Como extraños"
```

**💡 ¿Qué es?** Qué tan sincronizados se mueven los returns de ambos activos  
**💡 Analogía:** ¿Bailan al mismo ritmo o cada uno hace lo suyo?

---

### **🎯 CALIDAD DE SEÑALES**

#### **Signal Frequency**
```
🟩 > 10%    "Alta actividad - Muchas oportunidades"
🟨 > 5%     "Actividad moderada - Oportunidades suficientes"
🟥 ≤ 5%     "Baja actividad - Pocas oportunidades"
```

**💡 ¿Qué es?** % del tiempo que hay señales de trading  
**💡 Ejemplo:** 12% = "12 de cada 100 períodos hay oportunidad"

#### **Average Duration (Convertido a Horas)**
```
🟩 > 2h     "Duración cómoda - Tiempo para ejecutar"
🟨 > 0.5h   "Duración corta - Ejecución rápida"
🟥 ≤ 0.5h   "Muy corta - Difícil de ejecutar"
```

**💡 ¿Qué es?** Tiempo promedio que duran las señales  
**💡 Ejemplo:** 1.6h = "En promedio, mantienes posición 1.6 horas"

#### **False Signal Rate**
```
🟩 < 5%     "Excelente calidad - Señales muy confiables"
🟨 < 10%    "Buena calidad - Señales confiables"
🟥 ≥ 10%    "Calidad cuestionable - Muchas señales falsas"
```

**💡 ¿Qué es?** % de señales que se revierten rápidamente  
**💡 Ejemplo:** 0% = "Ninguna señal falsa - todas confiables"

#### **Signal Consistency**
```
🟩 > 80%    "Alta consistencia - Señales estables"
🟨 > 60%    "Consistencia moderada - Aceptable"
🟥 ≤ 60%    "Baja consistencia - Señales erráticas"
```

**💡 ¿Qué es?** Estabilidad de señales (menos cambios bruscos)  
**💡 Ejemplo:** 96% = "Señales muy estables, pocos cambios erráticos"

---

## **⚡ TRADING ZONES**

### **🟢 Neutral Zone (|Z| < 2)**
- **Significado:** Spread cerca de la media
- **Acción:** No hacer nada, esperar
- **Ideal:** 70-85% del tiempo

### **🟡 Trading Zone (2 ≤ |Z| < 4)**
- **Significado:** Divergencia significativa  
- **Acción:** Entrar en posición
- **Ideal:** 10-25% del tiempo

### **🔴 Extreme Zone (|Z| ≥ 4)**
- **Significado:** Divergencia extrema
- **Acción:** Stop-loss o salida emergencia
- **Ideal:** < 5% del tiempo

---

## **🎯 GUÍA DE DECISIÓN RÁPIDA**

### **🚀 TRADE INMEDIATO (Confianza 90%+)**
```
✅ 4+ métricas AZULES
✅ 0 métricas ROJAS  
✅ ADF < 0.01
✅ Cointegration Ratio ≥ 0.7
```

### **✅ TRADE CON CONFIANZA (Confianza 75-90%)**
```
✅ 2+ métricas AZULES
✅ 4+ métricas VERDES
✅ ≤1 métrica ROJA
✅ ADF < 0.05
```

### **⚠️ TRADE CON PRECAUCIÓN (Confianza 60-75%)**
```
⚠️ Mayoría métricas AMARILLAS
⚠️ ADF verde o mejor
⚠️ Sin más de 1 métrica ROJA
⚠️ Posición reducida (50-70%)
```

### **❌ NO TRADEAR (Confianza <60%)**
```
❌ 2+ métricas ROJAS
❌ ADF rojo (p ≥ 0.10)
❌ Cointegration Ratio < 0.4
❌ False signals ≥ 15%
```

---

## **🔧 CHECKLIST DE EVALUACIÓN**

### **📋 Pair EXCELENTE**
- [ ] Cointegración: Ratio ≥ 0.7
- [ ] ADF: p < 0.01
- [ ] KPSS: p > 0.02  
- [ ] Half-life: 6-14% del window
- [ ] R²: > 60%
- [ ] Correlación: > 0.45
- [ ] False signals: < 5%
- [ ] Signal frequency: > 10%

### **📋 Pair ACEPTABLE** 
- [ ] Cointegración: Ratio ≥ 0.4
- [ ] ADF: p < 0.05
- [ ] KPSS: p > 0.005
- [ ] Half-life: < 45% del window
- [ ] R²: > 40%
- [ ] Correlación: > 0.25
- [ ] False signals: < 10%
- [ ] Signal frequency: > 5%

### **📋 Pair RECHAZADO**
- [ ] Cointegración: Ratio < 0.4
- [ ] ADF: p ≥ 0.10
- [ ] Correlación: ≤ 0.20
- [ ] R²: ≤ 40%
- [ ] False signals: ≥ 15%

---

## **💡 DIFERENCIAS CLAVE: TRADICIONAL vs MEMECOINS**

| Métrica | Tradicional | Memecoins | Factor |
|---------|-------------|-----------|--------|
| **Cointegration** | ≥ 0.8 | ≥ 0.4 | 2x más relajado |
| **KPSS p-value** | > 0.10 | > 0.005 | 20x más relajado |
| **Beta Stability** | < 0.10 | < 0.45 | 4.5x más relajado |
| **Signal Duration** | > 4h | > 0.5h | 8x más rápido |

**🎯 Filosofía:** *No buscar perfección académica, sino equilibrio práctico entre oportunidad y riesgo en alta volatilidad*

---

## **⚡ CONFIGURACIÓN TRADING**

### **📊 Setup Básico**
```
ENTRY: |Z_robust| > 2.0
EXIT: Z crosses 0.0  
STOP: |Z_robust| > 4.0
```

### **💰 Gestión Posición**
```
AZUL/VERDE: 100% posición objetivo
AMARILLO: 50-70% posición objetivo  
ROJO: 0% (no tradear)
```

### **⏰ Monitoreo**
```
SEÑALES RÁPIDAS (<2h): Cada 15-30min
SEÑALES LARGAS (>4h): Cada 1-2h
MÁXIMO CAPITAL: 5-10% por pair
```

---

