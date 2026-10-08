from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parent
CSV = ROOT / 'dados/simulacao_indicadores_economicos_brasil.csv'
LABELS = {'pib':'PIB (unidade da base)', 'inflacao':'Inflação (%)', 'taxa_juros':'Juros (%)', 'taxa_desemprego':'Desemprego (%)', 'cambio_dolar':'Câmbio (R$/US$)', 'renda_media':'Renda média (R$)', 'consumo_familias':'Consumo (índice)', 'investimento':'Investimento (índice)', 'exportacoes':'Exportações (unidade da base)', 'importacoes':'Importações (unidade da base)'}
NUM = list(LABELS)

def preparar(origem=CSV):
    df = pd.read_csv(origem)
    obrigatorias = ['ano','trimestre','data',*NUM,'nivel_economico']
    faltam = set(obrigatorias) - set(df.columns)
    if faltam:
        raise ValueError(f'Colunas ausentes: {sorted(faltam)}')
    df = df[obrigatorias].copy()
    total = len(df)
    duplicados = int(df.duplicated().sum())
    df = df.drop_duplicates()
    df['data'] = pd.to_datetime(df['data'], errors='coerce')
    for col in ['ano','trimestre',*NUM]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.replace([np.inf,-np.inf],np.nan)
    df['nivel_economico'] = df['nivel_economico'].astype('string').str.strip().str.capitalize()
    invalidos = df[obrigatorias].isna().any(axis=1)
    invalidos |= (df['ano'] != df['data'].dt.year) | (df['trimestre'] != df['data'].dt.quarter)
    invalidos |= ~df['nivel_economico'].isin(['Crise','Estabilidade','Crescimento'])
    invalidos |= (df[['pib','cambio_dolar','renda_media','consumo_familias','investimento','exportacoes','importacoes']] <= 0).any(axis=1)
    invalidos |= ~df['taxa_desemprego'].between(0,100)
    removidos = int(invalidos.sum())
    df = df.loc[~invalidos].sort_values('data').reset_index(drop=True)
    if df.empty:
        raise ValueError('Nenhuma observação válida após a preparação.')
    if df.duplicated(['ano','trimestre']).any():
        raise ValueError('Há registros conflitantes para o mesmo trimestre; corrija a fonte.')
    df[['ano','trimestre']] = df[['ano','trimestre']].astype(int)
    df['periodo'] = df['ano'].astype(str) + '-T' + df['trimestre'].astype(str)
    # Reindexar preserva lacunas e impede comparar trimestres não consecutivos.
    idx = pd.PeriodIndex(df['data'],freq='Q')
    serie = pd.Series(df['pib'].to_numpy(),index=idx)
    completa = serie.reindex(pd.period_range(idx.min(),idx.max(),freq='Q'))
    df['crescimento_pib_pct'] = (completa.pct_change(fill_method=None)*100).reindex(idx).to_numpy()
    df['pib_media_movel_4t'] = completa.rolling(4,min_periods=4).mean().reindex(idx).to_numpy()
    df['saldo_comercial'] = df['exportacoes'] - df['importacoes']
    relatorio = {'linhas_originais':total,'duplicatas_removidas':duplicados,'linhas_invalidas_removidas':removidos,'linhas_validas':len(df),'trimestres_ausentes':int(completa.isna().sum())}
    return df, relatorio

def persistir(df, caminho=None):
    caminho = Path(caminho or ROOT / 'database/indicadores.db')
    caminho.parent.mkdir(parents=True,exist_ok=True)
    engine = create_engine('sqlite:///' + caminho.as_posix())
    try:
        with engine.begin() as conn:
            df.to_sql('indicadores',conn,if_exists='replace',index=False)
            conn.execute(text('CREATE UNIQUE INDEX IF NOT EXISTS idx_periodo ON indicadores(ano,trimestre)'))
        with engine.connect() as conn:
            resultado = pd.read_sql(text('SELECT ano, COUNT(*) AS trimestres, AVG(pib) AS pib_medio, AVG(inflacao) AS inflacao_media FROM indicadores GROUP BY ano ORDER BY ano'),conn)
    finally:
        engine.dispose()
    return resultado

def conclusao(df):
    if df.empty:
        return 'Não há observações no recorte.'
    maior = df.loc[df.pib.idxmax()]
    cv = df[NUM].std().div(df[NUM].mean().abs()).dropna()*100
    instavel = LABELS[cv.idxmax()] if not cv.empty else 'indisponível'
    r1 = df.inflacao.corr(df.taxa_desemprego) if len(df)>2 else float('nan')
    r2 = df.taxa_juros.corr(df.consumo_familias) if len(df)>2 else float('nan')
    def fmt(x):
        return f'{x:.2f}' if pd.notna(x) else 'indisponível'
    return (f'No recorte de {len(df)} trimestres, o maior PIB ocorre em {maior.periodo}. '
            f'A correlação inflação–desemprego é {fmt(r1)} e juros–consumo é {fmt(r2)}. '
            f'O maior coeficiente de variação dos níveis é de {instavel}. '
            f'Há {int(df.nivel_economico.eq("Crise").sum())} registros rotulados como Crise pela fonte. '
            'Essas associações não demonstram causalidade. A base é simulada, os rótulos não equivalem a uma classificação oficial de recessão e não permitem diagnosticar a economia brasileira real.')
