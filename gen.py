import random

import numpy as np

from jogo import minimax, inicializar_tabuleiro, verificar_estado
import RedeNeural


def gen():
    tamanho_populacao = 100
    num_pesos = 30  # Exemplo: Ajustar para a topologia da sua rede
    rede = RedeNeural(entrada=9, ocultos=12, saida=9)

    for i in range(300):
        print("Geração: " + str(i + 1))
        # jogar
        # analisa pela função heuristica
        # verifica se achou o melhor
        # aplica elitismo
        # aplica torneio
        # cruzamento
        # mutação


def inicializar_populacao(tamanho, num_pesos):
    # Inicializa a população com valores aleatórios
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
            entrada_rede = np.array(tabuleiro).flatten()
            saidas = rede.forward(entrada_rede)

            # Converte a saída em uma posição do tabuleiro
            indices_ordenados = np.argsort(-saidas)  # Ordena pela probabilidade
            jogada_valida = False

            for idx in indices_ordenados:
                i, j = divmod(idx, 3)  # Converte índice linear para linha e coluna
                if realizar_jogada(tabuleiro, jogador_rede, (i, j)):
                    jogada_valida = True
                    break

            if not jogada_valida:
                pontuacao -= 10
                break

        else:  # Minimax decide a jogada

            usar_minimax = False

            if dificuldade == 1:
                usar_minimax = random.random() < 0.25
            elif dificuldade == 2:
                usar_minimax = random.random() < 0.50
            elif dificuldade == 3:
                usar_minimax = True

            if usar_minimax:
                posicao = minimax(tabuleiro, 0, True, jogador_minimax, jogador_rede)

            else:
                posicao = random.random() < 0.5

            if posicao:
                jogadas_possiveis = [(i, j) for i in range(3) for j in range(3) if tabuleiro[i][j] == ' ']
                if jogadas_possiveis:
                    linha, coluna = random.choice(jogadas_possiveis)
                    tabuleiro[linha][coluna] = jogador_minimax

        # Verifica o estado do jogo após cada jogada
        estado = verificar_estado(tabuleiro)
        if estado:
            if estado == jogador_rede:
                pontuacao += 10  # Vitória da rede
            elif estado == jogador_minimax:
                pontuacao -= 10  # Derrota da rede
            else:
                pontuacao += 1  # Empate
            jogando = False  # Fim do jogo
        else:
            # Alterna o jogador
            jogador_atual = "Minimax" if jogador_atual == "Rede" else "Rede"

    return pontuacao




def avaliar_cromossomo(cromossomo, rede, dificuldade):
    # Configura a rede neural com os pesos do cromossomo
    rede.set_pesos(cromossomo)

    # Joga contra o Minimax e calcula pontuação (heurística)
    pontuacao = 0
    for _ in range(5):  # Exemplo: 5 partidas por cromossomo
        resultado = jogar_partida(rede, dificuldade)
        pontuacao += resultado
    return pontuacao


def aplicar_selecao_elitismo(populacao, avaliacoes, num_elitistas=2):
    # Mantém os melhores cromossomos
    melhores_indices = np.argsort(avaliacoes)[-num_elitistas:]
    return [populacao[i] for i in melhores_indices]


def aplicar_torneio_cruzamento(populacao, tamanho_torneio=3):
    nova_populacao = []
    while len(nova_populacao) < len(populacao):
        pais = random.sample(populacao, tamanho_torneio)
        pai = max(pais, key=lambda x: x[1])
        mae = min(pais, key=lambda x: x[1])
        filho = cruzamento(pai[0], mae[0])
        nova_populacao.append(filho)
    return nova_populacao


def cruzamento(pai, mae):
    # Cruzamento de um ponto
    ponto_corte = random.randint(1, len(pai) - 1)
    return np.concatenate((pai[:ponto_corte], mae[ponto_corte:]))


def mutacao(cromossomo, taxa_mutacao=0.01):
    # Altera os pesos aleatoriamente com uma certa probabilidade
    for i in range(len(cromossomo)):
        if random.random() < taxa_mutacao:
            cromossomo[i] += np.random.uniform(-0.5, 0.5)
    return cromossomo
