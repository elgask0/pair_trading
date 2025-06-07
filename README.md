# Sistema de Trading Cuantitativo de Pares

Sistema de pair trading con análisis de liquidez de alta frecuencia, cálculo de mark prices VWAP, y gestión automática de calidad de datos.

## 🎯 Características Principales

- **Pair Trading**: Estrategias de mean reversion basadas en Z-score
- **Mark Prices VWAP**: Precios realistas usando orderbook completo
- **Análisis de Liquidez**: Slippage detallado por tamaño de orden
- **Calidad de Datos**: Sistema P80 automático de clasificación
- **Funding Rates**: Análisis de costos para contratos perpetuos

## 📊 Arquitectura de Base de Datos

### Tablas Principales

#### `ohlcv` - Datos de Precio/Volumen
```sql
symbol VARCHAR(50)           -- Instrumento (ej: "MEXCFTS_PERP_GIGA_USDT")
timestamp TIMESTAMP          -- Marca temporal
open/high/low/close FLOAT    -- Precios OHLC
volume FLOAT                 -- Volumen
```

#### `orderbook` - Snapshots L2 del Orderbook
```sql
symbol VARCHAR(50)
timestamp TIMESTAMP
bid1_price, bid1_size FLOAT  -- Mejor bid
ask1_price, ask1_size FLOAT  -- Mejor ask
bid2_price, bid2_size FLOAT  -- Hasta 10 niveles por lado
...
-- Campos de calidad (agregados por scripts)
liquidity_quality VARCHAR   -- "Excellent", "Good", "Fair", "Poor", "Invalid"
valid_for_trading BOOLEAN   -- TRUE solo para Excellent/Good
spread_pct FLOAT            -- Spread calculado
threshold_p80 FLOAT         -- Umbral P80 para clasificación
```

#### `mark_prices` - Precios VWAP Calculados
```sql
symbol VARCHAR(50)
timestamp TIMESTAMP
mark_price FLOAT            -- Precio VWAP del orderbook
orderbook_mid FLOAT         -- (bid1+ask1)/2 para comparación
ohlcv_close FLOAT          -- OHLCV close para validación
bid_ask_spread_pct FLOAT   -- Spread VWAP ponderado
liquidity_score FLOAT      -- Score de calidad (0-1)
is_valid BOOLEAN           -- Flag de validación
validation_source VARCHAR  -- Fuente de validación
```

#### `funding_rates` - Tasas de Financiamiento
```sql
symbol VARCHAR(50)          -- Solo contratos "PERP_"
timestamp TIMESTAMP
funding_rate FLOAT          -- Tasa como decimal (0.0001 = 0.01%)
collect_cycle INTEGER       -- Período en segundos (28800 = 8h)
```

#### `pair_configurations` - Configuración de Estrategias
```sql
pair_name VARCHAR(100)      -- "GIGA_SPX"
symbol1, symbol2 VARCHAR    -- Instrumentos del par
entry_zscore FLOAT         -- Z-score para entrada (default: 2.0)
exit_zscore FLOAT          -- Z-score para salida (default: 0.5)
stop_loss_zscore FLOAT     -- Z-score para stop (default: 3.0)
position_size_pct FLOAT    -- % del capital (default: 0.1)
zscore_window INTEGER      -- Ventana en minutos (default: 60)
min_correlation FLOAT      -- Correlación mínima (default: 0.7)
is_active BOOLEAN          -- Activo para trading
```

## 🔧 Scripts Principales

### `scripts/setup_database.py` - Inicialización
- Crea todas las tablas y esquemas
- Configura índices optimizados
- Puebla configuraciones iniciales desde `config/symbols.yaml`

### `scripts/ingest_data.py` - Ingesta de Datos
```bash
# Ingesta completa
python scripts/ingest_data.py

# Solo funding rates
python scripts/ingest_data.py --funding-only

# Símbolo específico
python scripts/ingest_data.py --symbol MEXCFTS_PERP_GIGA_USDT
```
- Descarga OHLCV, orderbook L2 y funding rates desde CoinAPI
- Ingesta incremental (solo datos faltantes)
- Validación en tiempo real

### `scripts/validate_data.py` - Validación de Integridad
- **OHLCV**: Verifica relaciones OHLC, volúmenes válidos
- **Orderbook**: Detecta spreads cruzados, precios inválidos
- **Completitud**: Identifica gaps temporales
- Genera reporte detallado de problemas

### `scripts/clean_data.py` - Limpieza y Calidad
**Algoritmo P80**:
1. Calcula percentil 80 de spreads válidos por símbolo
2. Clasifica calidad:
   - `Excellent`: ≤50% del P80
   - `Good`: 50% < spread ≤ P80  
   - `Fair`: P80 < spread ≤ 150% del P80
   - `Poor`: >150% del P80
3. Marca `valid_for_trading = TRUE` solo para Excellent/Good
4. **Preserva todos los datos** (no elimina nada)

### `scripts/calculate_markprices.py` - Mark Prices VWAP
**Algoritmo**:
1. Extrae hasta 10 niveles del orderbook
2. Filtra niveles con volumen <$1,000
3. Calcula VWAP separado para bid/ask
4. Combina usando ponderación por liquidez
5. Valida contra rango OHLCV
6. Genera quality score (0-1) basado en:
   - Profundidad del orderbook (30%)
   - Volumen total (30%)
   - Spread VWAP (30%)
   - Balance liquidez (10%)

### `scripts/analyze_liquidity.py` - Análisis de Liquidez
- Simula órdenes de $100-$5,000
- Calcula slippage por tamaño
- Mide tasas de ejecución exitosa
- Genera grading A-D de calidad

### `scripts/analyze_data.py` - Análisis Avanzado
- Análisis bidireccional (compra/venta)
- Box plots de distribuciones de slippage
- Órdenes de $100-$10,000
- Market impact y consumo de liquidez
- Usa TODOS los datos disponibles (no solo 7 días)

## ⚙️ Configuración

### `.env` - Variables de Entorno
```env
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=pair_trading
DB_USER=postgres
DB_PASSWORD=your_password

# CoinAPI
COINAPI_KEY=your_coinapi_key

# Capital
INITIAL_CAPITAL=10000.0
```

### `config/symbols.yaml` - Pares de Trading
```yaml
trading_pairs:
  - symbol1: "MEXCFTS_PERP_GIGA_USDT"
    symbol2: "MEXCFTS_PERP_SPX_USDT"
    entry_zscore: 2.0
    exit_zscore: 0.5
    stop_loss_zscore: 3.0
    position_size_pct: 0.1
    zscore_window: 60
    min_correlation: 0.7
    is_active: true
```

## 🚀 Flujo de Trabajo

### 1. Setup Inicial
```bash
# Configurar .env y symbols.yaml
python scripts/setup_database.py
```

### 2. Ingesta de Datos
```bash
# Funding rates (rápido para testing)
python scripts/ingest_data.py --funding-only

# Datos completos
python scripts/ingest_data.py
```

### 3. Procesamiento
```bash
# Validar integridad
python scripts/validate_data.py

# Limpiar y marcar calidad
python scripts/clean_data.py

# Calcular mark prices VWAP
python scripts/calculate_markprices.py
```

### 4. Análisis
```bash
# Análisis de liquidez básico
python scripts/analyze_liquidity.py

# Análisis avanzado con distribuciones
python scripts/analyze_data.py
```

## 📈 Consultas SQL Útiles

### Datos de Alta Calidad para Backtesting
```sql
-- Orderbook trading-ready
SELECT * FROM orderbook 
WHERE valid_for_trading = TRUE
AND liquidity_quality IN ('Excellent', 'Good')
ORDER BY symbol, timestamp;

-- Mark prices validados
SELECT * FROM mark_prices 
WHERE is_valid = TRUE 
AND liquidity_score > 0.7
ORDER BY symbol, timestamp;
```

### Análisis de Correlación
```sql
WITH returns AS (
    SELECT 
        symbol,
        timestamp,
        mark_price / LAG(mark_price) OVER (PARTITION BY symbol ORDER BY timestamp) - 1 as return_pct
    FROM mark_prices 
    WHERE is_valid = TRUE
)
SELECT 
    a.timestamp,
    a.return_pct as symbol1_return,
    b.return_pct as symbol2_return,
    a.return_pct - b.return_pct as spread_return
FROM returns a
JOIN returns b ON a.timestamp = b.timestamp
WHERE a.symbol = 'MEXCFTS_PERP_GIGA_USDT'
AND b.symbol = 'MEXCFTS_PERP_SPX_USDT';
```

## 🔍 Testing y Debugging

```bash
# Verificar funcionalidad de mark prices
python scripts/test_markprice.py

# Probar ingesta para fecha específica
python scripts/test_ingestion.py --date 2024-01-15

# Forzar recálculo si hay problemas
python scripts/clean_data.py --force
python scripts/calculate_markprices.py --force
```

## 📝 Algoritmos Clave

### Sistema P80 de Calidad
1. Calcula `P80 = percentile_80(spreads_válidos_por_símbolo)`
2. Clasifica: `Excellent` (≤50% P80), `Good` (≤P80), `Fair` (≤150% P80), `Poor` (>150% P80)
3. Solo `Excellent` y `Good` → `valid_for_trading = TRUE`

### VWAP Mark Price
1. Filtra niveles orderbook por volumen mínimo ($1K)
2. VWAP_bid = Σ(price×size) / Σ(size) para lado bid
3. VWAP_ask = Σ(price×size) / Σ(size) para lado ask  
4. Mark_price = (VWAP_bid × weight_bid) + (VWAP_ask × weight_ask)
5. Weights basados en liquidez disponible

### Generación de Señales Z-Score
1. Spread = log(price1 / price2)
2. Z-score = (spread - rolling_mean) / rolling_std
3. Señales preeliminares: Entry (|Z| > 2.0), Exit (|Z| < 0.5), Stop (|Z| > 4.0)

El sistema está optimizado para análisis cuantitativo riguroso y backtesting preciso con costos de ejecución realistas.