#---------------------Importando os módulos------------------------#
import cv2
import numpy

#---------------Imagens Geradas com as Primeiras K Colunas de U, V e K Valores Singulares------------------#
def ImgK(U, S, V, K, M, Porcentagem): #Porcentagem = Porcentagem obtida para o K
	Ur = [] #Matriz U reduzida
	Sr = [] #Matriz Sigma reduzida
	Vr = [] #Matriz V reduzida
	for i in range(K): #Pegando as K primeiras colunas de U para compor Ur
		Uaux = []
		for j in range(512):
			Uaux.append(U[j][i])
		Ur.append(Uaux)
	Ur = numpy.array(Ur)
	Ur = Ur.T
	#print(Ur)
	for i in range(K): #Pegando as K primeiros valores singulares de S para compor Sr
		Sr.append(S[i])
	Sr = numpy.diagflat(Sr) #Transformando o vetor de valores sing em uma matriz diagonal
	#print(Sr)
	for i in range(K): #Pegando as K primeiros colunas de V para compor Vr
		Vaux = []
		for j in range(512):
			Vaux.append(V[j][i])
		Vr.append(Vaux)
	Vr = numpy.array(Vr)
	#print(Vr)
	Xr = Ur.dot(Sr).dot(Vr) #Multiplicando as matrizes
	for i in range(512): #Somando as médias respectivas de cada coluna
		for j in range(512):
			Xr[j][i] += M[i] 
	#print(Xr)
	nome = "lena_" + Porcentagem + ".jpg" #Fomatando nome para salvar e exibir a imagem
	cv2.imshow(nome, Xr)
	cv2.waitKey(0)
	Xr = (Xr * 255).astype(numpy.uint8)
	cv2.imwrite(nome, Xr) #Salva imagem do imshow()

#---------------Imagens Geradas com as Ultimas K Colunas de U, V e Valores Singulares------------------#
def Img20ultimos(U, S, V, K, M):
	Ur20 = [] #Matriz U reduzida
	Sr20 = [] #Matriz Sigma reduzida
	Vr20 = [] #Matriz V reduzida
	for i in range(K, 512): #Pegando as K ultimas colunas de U para compor Ur20
		Uaux = []
		for j in range(512):
			Uaux.append(U[j][i])
		Ur20.append(Uaux)
	Ur20 = numpy.array(Ur20)
	Ur20 = Ur20.T
	#print(Ur20)
	for i in range(K, 512): #Pegando as K ultimos valores singulares de S para compor Sr20
		Sr20.append(S[i])
	Sr20 = numpy.diagflat(Sr20) 
	#print(Sr20)
	for i in range(K, 512):	 #Pegando as K primeiros colunas de V para compor Vr
		Vaux = []
		for j in range(512):
			Vaux.append(V[j][i])
		Vr20.append(Vaux)
	Vr20 = numpy.array(Vr20)
	#print(Vr20)
	Xr20 = Ur20.dot(Sr20).dot(Vr20)
	for i in range(512):
		for j in range(512):
			Xr20[j][i] += M[i] 
	#print(Xr20)
	cv2.imwrite("lena_E20_ultimosVS.jpg", Xr20) #Salva imagem
	img2 = cv2.imread("lena_E20_ultimosVS.jpg", -1) #Gambiarra
	for i in range(512):
		for j in range(512):
			if img2[i][j] == 1:
				img2[i][j] += 5
	cv2.imshow("lena_E20_ultimosVS.jpg", img2)
	cv2.waitKey(0)

	"""O imshow() gera imagens a partir de valores entre 0 e 1. Já, o imwrite() gera imagens a partir de
	valores entre 0 e 255. Então é necessarios fazer essa "gambiarra", a fim de que a imagem mostrada
	no imshow() seja iguail a salva com o imwrite(). A gambiarra carrega a imagem com valores entre 0 e 255
	salva anteriormente, e posteriormente, soma o valor 5 a todos os 1's da imagem para que o imshow() 
	passe a interpretar a imagem com valores entre 0 e 255."""

#-------------------Verificação da Igualdade---------------------#
"""*O professor pediu por email que fizessemos essa verificação*
Verificando se os valores singulares de SVD ao quadrado são 
iguais aos autovalores da Decomposição Espectral."""
def Verificação(Lambda, S):
	S = S ** 2 #Elevando valores singulares ao quadrado 
	for i in range(512):	#Arrendondando valores de S e Lambda com 5 casas decimais
		Lambda[i] = round(Lambda[i], 5)
		S[i] = round(Lambda[i], 5)
	comparação = S == Lambda
	result = comparação.all() #Verificando se os elementos são iguais
	return result

#------------Carregando Imagem Sem Qualquer Alteração--------------#
Y = cv2.imread("lena_gray.jpg", -1) #Matriz(numpy.array) que representa a imagem lena_gray.jpg
print("Matriz Antes de Qualquer Alteração:")
print(Y, "\n")

#--------------Dividindo Todos Os Elementos Por 255----------------#
Y = numpy.divide(Y, 255) #Divide cada elemento da matriz por 255
print("Dividindo Todos os Elementos Por 255:")
print(Y, "\n")

#---------------Calculando As Médias De Cada Coluna----------------#
soma = numpy.sum(Y, axis=0) #axis 0 significa que as colunas serão somadas
M = soma.copy() #numpy array com 1 dimensão de tamanho 512. Cada elem é a soma da respectiva coluna
M = numpy.divide(M, 512) #Divide cada elemento do numpy array por 512
#print(M)

#---------Subtraindo Cada Elemento Da Coluna Pela Respectiva Média--------#
for i in range(512):
    for j in range(512):
        Y[j][i] -= M[i]
X = Y.copy() #Renomeando a Matriz
print("Matriz Após Alterações:")
print(X, "\n")

#---------Fazendo a Decomposição SVD da Matriz X------------#
U, S, V = numpy.linalg.svd(X)  #O S é um array numpy de 1 dimensão
#O V obtido na decomposição está transposto, devemos transpô-lo novamente nas chamadas das funções
S.sort() 
S = S[::-1] #S em ordem decrescente
'''
S = numpy.diagflat(S) #Transforma o array em uma matriz diagonal
SVD = U.dot(S).dot(V)
print(SVD) #Verificar se a decomposição está correta. SVD tem que ser igual a X(Matriz após alterações)
'''

#-----------------Testando Valores Para K--------------------#
Ke020, Ke090, Ke099 = None, None, None
numerador = 0 
denominador = 0
for i in range(512): #calculando o denominador s^2(0)+s^2(2)+...+s^2(511)
	denominador += S[i]**2
print("Valores de E:")
for k in range(512):
	numerador += S[k]**2
	E = numerador/denominador
	#print(E)
	if E < 0.24: #Quando E < 24% Ke020 recebe o valor de K + 1, já que K começa em 0
		Ke020 = k+1 
	elif E > 0.90 and E < 0.91: #Quando E < 90% Ke090 recebe o valor de K + 1, já que K começa em 0
		Ke090 = k+1
	elif E > 0.99: #Quando E < 99% Ke020 recebe o valor de K + 1, já que K começa em 0
		Ke099 = k+1
		break #É usado pois existem vários valores entre 99% e 100%. Pega apenas a primeira aparição e termina o for 
print(f"Para k[E=0.20] = {Ke020};\nPara k[E=0.90] = {Ke090};\nPara k[E=0.99] = {Ke099}.")

#---------------Chamadas as Funções Geradora das Imagens----------------#
"""Como mensionado anteriormente, devemos transpor V *(V.T).T = V*,
pois queremos as k colunas de V,e não de V transposto(Resultado da Decomposição SVD)"""
ImgK(U, S, V.T, Ke020, M, "20") #Imagem com E = 20%
ImgK(U, S, V.T, Ke090, M, "90") #Imagem com E = 90%
ImgK(U, S, V.T, Ke099, M, "99") #Imagem com E = 99%
Img20ultimos(U, S, V.T, Ke020, M) #Imagem formada com os 512-k últimos valores singulares

#---------Decomposição Espectral e Chamada da Função de Verificação de Igualdade----------#
A = numpy.dot(X.T, X)
Lambda, Q = numpy.linalg.eig(A)
Lambda.sort()
Lambda = Lambda[::-1]
print(f"Os Autovalores de Lambda são o quadrado dos Valores Singulares de S?\n{Verificação(Lambda, S)}") 
#retorna True se a Resposta for sim

#----------------------------Fim--------------------------------------#
