from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from analise import CSV, LABELS, NUM, preparar, persistir, conclusao

st.set_page_config(page_title='Observatório Econômico | Tema 17',page_icon='📊',layout='wide')
st.title('Observatório Econômico do Brasil')
st.caption('PROJETO G1 · TEMA 17 · 2015–2024')
st.markdown('**Aluna:** Lorrane Goulart  \n**Professor:** Alexandre Neves Louzada')
st.write('Como evoluem o PIB, os preços, a renda e o consumo? Explore indicadores trimestrais, compare períodos e investigue relações entre variáveis.')
st.info('Base simulada para fins acadêmicos. Os valores não representam estatísticas oficiais. As unidades e a periodicidade de acumulação do PIB e da inflação não foram detalhadas pela fonte.')

@st.cache_data
def carregar(conteudo):
    return preparar(BytesIO(conteudo))

base, qualidade = carregar(CSV.read_bytes())
with st.sidebar:
    st.header('Filtrar análise')
    anos = st.multiselect('Anos',sorted(base.ano.unique()),default=sorted(base.ano.unique()))
    trimestres = st.multiselect('Trimestres',[1,2,3,4],default=[1,2,3,4])
    niveis = st.multiselect('Nível econômico',sorted(base.nivel_economico.unique()),default=sorted(base.nivel_economico.unique()))
    indicadores = st.multiselect('Indicadores',NUM,default=['pib','inflacao','taxa_juros','taxa_desemprego'],format_func=LABELS.get)
    st.caption('Filtros temporais e de nível afetam KPIs e análises. A seleção de indicadores controla os gráficos comparativos; os dois pares de correlação pedidos permanecem fixos.')

df = base[base.ano.isin(anos)&base.trimestre.isin(trimestres)&base.nivel_economico.isin(niveis)].copy()
if df.empty:
    st.warning('Nenhum registro para os filtros selecionados. Selecione ao menos um ano, trimestre e nível econômico.')
    st.stop()

kpis = [('PIB médio',df.pib.mean(),'unid. da base'),('Inflação média',df.inflacao.mean(),'%'),('Desemprego médio',df.taxa_desemprego.mean(),'%'),('Juros médios',df.taxa_juros.mean(),'%'),('Câmbio médio',df.cambio_dolar.mean(),'R$/US$'),('Variação média do PIB',df.crescimento_pib_pct.mean(),'% t/t')]
for col,(nome,valor,unidade) in zip(st.columns(6),kpis):
    col.metric(nome,'N/D' if pd.isna(valor) else f'{valor:,.2f}'.replace(',','X').replace('.',',').replace('X','.'),help=unidade)
st.caption('Médias aritméticas das observações selecionadas. Variação do PIB calculada antes dos filtros contra o trimestre imediatamente anterior da base; não é crescimento real nem anualizado.')
visao,relacoes,comparacao,dados = st.tabs(['Evolução temporal','Relações e instabilidade','Comparação e radar','Dados e metodologia'])
with visao:
    if not indicadores:
        st.warning('Selecione indicadores na barra lateral para exibir os gráficos.')
    for indicador in indicadores:
        # Trimestres removidos permanecem como lacunas na linha.
        linha = base[['data']].merge(df[['data',indicador]],on='data',how='left')
        fig = px.line(linha,x='data',y=indicador,markers=True,labels={'data':'Trimestre',indicador:LABELS[indicador]},title=LABELS[indicador],template='plotly_white')
        st.plotly_chart(fig,width='stretch')
    st.subheader('PIB e média móvel de quatro trimestres')
    st.line_chart(df.set_index('data')[['pib','pib_media_movel_4t']])
    st.caption('A média móvel usa quatro trimestres consecutivos da base completa, incluindo os anteriores ao recorte. Não substitui ajuste sazonal.')
    taxas = df.dropna(subset=['crescimento_pib_pct'])
    if not taxas.empty:
        pico = taxas.loc[taxas.crescimento_pib_pct.idxmax()]
        st.write(f'Maior variação trimestral do PIB no recorte: {pico.crescimento_pib_pct:.2f}% em {pico.periodo}.')
    st.subheader('Períodos rotulados como crise')
    st.dataframe(df.loc[df.nivel_economico.eq('Crise'),['periodo','pib','crescimento_pib_pct','inflacao','taxa_desemprego']],hide_index=True)
    st.caption('O rótulo original foi preservado. Uma queda do PIB não necessariamente coincide com esse rótulo simulado.')
with relacoes:
    for x,y in [('inflacao','taxa_desemprego'),('taxa_juros','consumo_familias')]:
        st.plotly_chart(px.scatter(df,x=x,y=y,color='nivel_economico',hover_name='periodo',labels=LABELS,title=f'{LABELS[x]} × {LABELS[y]}'),width='stretch')
        r = df[x].corr(df[y]) if len(df)>2 else float('nan')
        st.write('Correlação de Pearson: '+(f'{r:.3f} (n={len(df)}).' if pd.notna(r) else 'indisponível; selecione mais observações com variação.'))
    st.caption('Correlação contemporânea, sem controles ou defasagens: não permite afirmar impacto causal dos juros sobre o consumo.')
    if len(df)>=3 and len(indicadores)>=2:
        fig,ax = plt.subplots(figsize=(9,5))
        sns.heatmap(df[indicadores].rename(columns=LABELS).corr(),annot=True,fmt='.2f',vmin=-1,vmax=1,cmap='vlag',ax=ax)
        ax.set_title('Matriz de correlação de Pearson')
        st.pyplot(fig);plt.close(fig)
    st.subheader('Instabilidade relativa dos níveis')
    cv = (df[NUM].std()/df[NUM].mean().abs()*100).sort_values(ascending=False)
    st.bar_chart(cv.rename(index=LABELS).rename('Coeficiente de variação (%)'))
    st.caption('CV = desvio-padrão amostral / |média| × 100. Mede dispersão dos níveis, não volatilidade dos retornos; tendências e médias próximas de zero afetam a comparação.')
with comparacao:
    if indicadores:
        ind = st.selectbox('Indicador para barras e heatmap',indicadores,format_func=LABELS.get)
        anual = df.groupby('ano',as_index=False)[ind].mean()
        st.plotly_chart(px.bar(anual,x='ano',y=ind,labels={ind:LABELS[ind],'ano':'Ano'},title='Média dos trimestres selecionados por ano'),width='stretch')
        matriz = df.pivot(index='ano',columns='trimestre',values=ind).reindex(columns=[1,2,3,4])
        fig,ax = plt.subplots(figsize=(10,5))
        sns.heatmap(matriz,annot=True,fmt='.1f',cmap='YlGnBu',ax=ax)
        ax.set_title(LABELS[ind]+' por ano e trimestre')
        st.pyplot(fig);plt.close(fig)
    if len(indicadores)>=3:
        minimo = base[indicadores].min()
        amplitude = (base[indicadores].max()-minimo).replace(0,float('nan'))
        radar = ((df[indicadores]-minimo)/amplitude*100).mean().fillna(0)
        st.plotly_chart(px.line_polar(r=radar.values,theta=[LABELS[x] for x in indicadores],line_close=True,range_r=[0,100],title='Radar das médias normalizadas (0–100)'),width='stretch')
        st.caption('Normalização mínimo–máximo pela base completa. Zero e 100 são os extremos observados. Maior valor não significa melhor desempenho. Variáveis constantes recebem zero.')
    else:
        st.info('Selecione ao menos três indicadores para o radar.')
    st.subheader('Tabela dinâmica')
    st.dataframe(df.pivot_table(index='ano',columns='nivel_economico',values=indicadores or ['pib'],aggfunc='mean'),width='stretch')
with dados:
    st.dataframe(df,hide_index=True,width='stretch')
    st.download_button('Baixar recorte em CSV',df.to_csv(index=False).encode('utf-8-sig'),'indicadores_filtrados.csv','text/csv')
    st.subheader('Qualidade da base')
    st.json(qualidade)
    st.write('São removidas duplicatas exatas e linhas inválidas; conflitos no mesmo trimestre impedem a carga. Dados faltantes não são inventados. Valores extremos válidos são preservados. Ano e trimestre são conferidos contra a data.')
    st.subheader('Dicionário dos indicadores')
    st.dataframe(pd.DataFrame({'coluna':NUM,'descrição / unidade':[LABELS[x] for x in NUM]}),hide_index=True)
    st.caption('As taxas usam a escala percentual informada no enunciado. Não se presume inflação acumulada anual ou juros efetivos trimestrais. Não há deflator para calcular renda ou PIB reais.')
    st.subheader('Persistência SQLite com SQLAlchemy')
    st.write('O botão grava a base tratada completa em database/indicadores.db e executa uma consulta SQL de médias anuais. Os filtros não alteram o banco.')
    if st.button('Gravar base e consultar banco'):
        st.dataframe(persistir(base),hide_index=True)
        st.success('Base gravada e consulta SQL executada.')
    st.caption('No Streamlit Community Cloud, o arquivo local pode ser recriado após reinício. O CSV versionado permite reconstruí-lo.')
st.subheader('Conclusão executiva do recorte')
st.write(conclusao(df))
