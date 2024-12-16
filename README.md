'''

A literatura mostra associação entre a copeptina alterada e a pré-eclâmpsia,
metadados referentes aos três trimestres selecionados a partir da seleção de trabalhos incluidos
em uma metanálise de subgrupo foram usados para treinar algoritmos de clusterização e de classificação 
para inferir a ocorrência da doença e ajudar na criação de um modelo de predição. 

Métodos de clusterização explorados com Scikit-learn do Python foram: 
KMeans, GaussianMixture, SpectralClustering, DBSCAN, Birch, MiniBatchKMeans, 
AgglomerativeClustering, OPTICS e MeanShift. 
Métricas para avaliação foram os scores:
Silhouett, Davies-Bouldin, Calinski-Harabasz, e Adjusted Rand.

Para a treinar os modelos de classificação foram usados os métodos de Random Forest, Extra Trees, e RNA,
avaliados pela matriz de confusão, acurácia, precisão, sensibilidade, AUC e especificidade.

'''
