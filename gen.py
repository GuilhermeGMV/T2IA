import random

import numpy as np

from jogo import minimax, inicializar_tabuleiro, jogar_contra_rede, verificar_estado
from RedeNeural import RedeNeural


def gen():
    tamanho_populacao = 10
    num_pesos = 237
    rede = RedeNeural(entrada=9, ocultos=12, saida=9)
    populacao = inicializar_populacao(tamanho_populacao, num_pesos)

    for geracao in range(300):
        print(f"Geração: {geracao + 1}")

        # Avaliação da população (população joga)
        avaliacoes = [avaliar_cromossomo(cromossomo, rede) for cromossomo in populacao]

        # Separar os dois melhores cromossomos
        melhores_indices = np.argsort(avaliacoes)[-2:]
        melhor_cromossomo1 = populacao[melhores_indices[-1]]
        melhor_cromossomo2 = populacao[melhores_indices[-2]]

        print(f"Melhor pontuação da geração {geracao + 1}: {avaliacoes[melhores_indices[-1]]}")
        print(f"Segunda melhor pontuação: {avaliacoes[melhores_indices[-2]]}")

        # Verificar se atingiu o fim
        if avaliacoes[melhores_indices[-1]] >= 60:
            print("Solução ótima encontrada!")
            print("Agora, jogue contra a rede!")
            rede.set_pesos(melhor_cromossomo1)

            with open("melhores.txt", "w") as arquivo:
                for peso in melhor_cromossomo1:
                    arquivo.write(f"{peso}\n")

            jogar_contra_rede(rede)
            break

        # Elitismo
        nova_populacao = [melhor_cromossomo1, melhor_cromossomo2]

        # Seleção, cruzamento e mutação nos demais cromossomos
        nova_populacao += torneio_cruzamento(
            populacao, avaliacoes
        )

        # Mutação
        nova_populacao = nova_populacao[:len(populacao)]
        print(sum(avaliacoes))

        # Muda a taxa de mutação de acordo com a avaliação geral
        taxa_mutacao = 0.01
        if sum(avaliacoes) == -1200:
            taxa_mutacao = 0.5
        elif sum(avaliacoes) >= -800:
            taxa_mutacao = 0.005

        nova_populacao = [mutacao(cromossomo, taxa_mutacao) for cromossomo in
                          nova_populacao[2:]]

        nova_populacao.insert(0, melhor_cromossomo1)
        nova_populacao.insert(1, melhor_cromossomo2)

        populacao = nova_populacao[:tamanho_populacao]


def inicializar_populacao(tamanho, num_pesos):
    return [np.random.uniform(-1, 1, num_pesos) for _ in range(tamanho)]


def jogar_partida(rede, dificuldade):
    tabuleiro = inicializar_tabuleiro()
    jogador_rede = 'X'
    jogador_minimax = 'O'
    jogador_atual = "Rede"
    pontuacao = 0

    def realizar_jogada(tabuleiro, jogador, posicao):
        i, j = posicao
        if tabuleiro[i][j] == ' ':
            tabuleiro[i][j] = jogador
            return True
        return False

    while True:
        if jogador_atual == "Rede":
            entrada_rede = np.array(
                [[1 if cell == 'X' else -1 if cell == 'O' else 0 for cell in row] for row in tabuleiro]).flatten()
            saidas = rede.forward(entrada_rede)

            indices_ordenados = np.argsort(-saidas)

            p = False

            for idx in indices_ordenados:
                i, j = divmod(idx, 3)
                if realizar_jogada(tabuleiro, jogador_rede, (i, j)):
                    break
                else:
                    pontuacao -= 20
                    p = True
                    break
            if p:
                break
        else:
            if dificuldade == 1:
                usar_minimax = random.random() < 0.25
            elif dificuldade == 2:
                usar_minimax = random.random() < 0.50
            else:
                usar_minimax = True

            if usar_minimax:
                melhor_pontuacao = float('inf')
                melhor_jogada = None

                for i in range(3):
                    for j in range(3):
                        if tabuleiro[i][j] == ' ':
                            tabuleiro[i][j] = jogador_minimax
                            pontuacao_minimax = minimax(tabuleiro, 0, True, jogador_minimax, jogador_rede)
                            tabuleiro[i][j] = ' '
                            if pontuacao_minimax < melhor_pontuacao:
                                melhor_pontuacao = pontuacao_minimax
                                melhor_jogada = (i, j)

                if melhor_jogada:
                    linha, coluna = melhor_jogada
                    tabuleiro[linha][coluna] = jogador_minimax
            else:
                jogadas_possiveis = [(i, j) for i in range(3) for j in range(3) if tabuleiro[i][j] == ' ']
                if jogadas_possiveis:
                    linha, coluna = random.choice(jogadas_possiveis)
                    tabuleiro[linha][coluna] = jogador_minimax

        estado = verificar_estado(tabuleiro)
        if estado:
            if estado == jogador_rede:
                pontuacao += 10  # Vitória da rede
            elif estado == jogador_minimax:
                pontuacao -= 5  # Derrota da rede
            else:
                pontuacao += 1  # Empate
            break
        else:
            jogador_atual = "Minimax" if jogador_atual == "Rede" else "Rede"
    return pontuacao


def avaliar_cromossomo(cromossomo, rede):
    rede.set_pesos(cromossomo)

    pontuacao = 0
    print('--------------------------------------------')
    for _ in range(6):
        resultado = jogar_partida(rede, (_ + 1) // 2)
        pontuacao += resultado
    print(pontuacao)
    return pontuacao


def elitismo(populacao, avaliacoes, num_elitistas=2):
    melhores_indices = np.argsort(avaliacoes)[-num_elitistas:]
    return [populacao[i] for i in melhores_indices]


def torneio_cruzamento(populacao, avaliacoes):
    populacao_ordenada = sorted(zip(avaliacoes, populacao), key=lambda x: x[0], reverse=True)

    nova_populacao = []

    while len(nova_populacao) < len(populacao):
        pai = populacao_ordenada[0][1]
        mae = populacao_ordenada[1][1]
        filho = cruzamento(pai, mae)
        nova_populacao.append(filho)
        if sum(avaliacoes) >= -1000:
            pai = populacao_ordenada[2][1]
            mae = populacao_ordenada[3][1]
            filho = cruzamento(pai, mae)
            nova_populacao.append(filho)

    return nova_populacao


def cruzamento(pai, mae):
    ponto_corte = random.randint(1, len(pai) - 1)
    return np.concatenate((pai[:ponto_corte], mae[ponto_corte:]))


def mutacao(cromossomo, taxa_mutacao=0.01):
    for i in range(len(cromossomo)):
        if random.random() < taxa_mutacao:
            cromossomo[i] += np.random.uniform(-0.5, 0.5)
    return cromossomo


def carregar_e_jogar(nome_arquivo, rede):
    try:
        with open(nome_arquivo, "r") as arquivo:
            pesos = [float(linha.strip()) for linha in arquivo]

        split1 = rede.pesos_entrada_oculta.size
        split2 = split1 + rede.pesos_oculta_saida.size
        split3 = split2 + rede.bias_oculta.size

        rede.pesos_entrada_oculta = np.array(pesos[:split1]).reshape(rede.pesos_entrada_oculta.shape)
        rede.pesos_oculta_saida = np.array(pesos[split1:split2]).reshape(rede.pesos_oculta_saida.shape)
        rede.bias_oculta = np.array(pesos[split2:split3])
        rede.bias_saida = np.array(pesos[split3:])

        print("Pesos carregados com sucesso! Iniciando o jogo contra a rede...")
        jogar_contra_rede(rede)

    except FileNotFoundError:
        print(f"Erro: O arquivo '{nome_arquivo}' não foi encontrado.")
    except Exception as e:
        print(f"Erro ao carregar os pesos: {e}")

