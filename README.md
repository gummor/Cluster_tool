'''
Visto que os relatos da literatura mostram associação entre a copeptina alterada e pré-eclâmpsia,
metadados referentes aos três trimestres a partir da seleção de trabalhos incluidos em 
uma revisão sistemática e metanálise de subgrupo foram usados para treinar e analisar
modelos de agrupamento e de classificação a fim de inferir a ocorrência da doença e 
ajudar na criação de um modelo de predição. 

Os métodos de classificação foram explorados usando a ferramenta Scikit-learn do Python: 
KMeans, GaussianMixture, SpectralClustering, DBSCAN, Birch, MiniBatchKMeans, 
AgglomerativeClustering, OPTICS e MeanShift. 
As métricas para avaliação foram os scores:
Silhouett, Davies-Bouldin, Calinski-Harabasz, e Adjusted Rand.

Para a treinar os modelos de classificação foram usados os métodos de Árvore de Decisão e de Redes Neurais,
avaliados pela matriz de confusão, acurácia, precisão, sensibilidade, AUC e especificidade.
'''
