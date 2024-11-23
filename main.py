from RedeNeural import RedeNeural
from gen import gen, carregar_e_jogar
from jogo import jogar_jogo

if __name__ == '__main__':
    while True:
        i = int(input('Digite 1 para jogar contra a melhor rede\nDigite 2 para rodar o AG\nDigite 3 para jogar contra o Minimax\nDigite 0 para Sair: '))
        if i == 0:
            break
        elif i == 1:
            rede = RedeNeural(entrada=9, ocultos=12, saida=9)
            carregar_e_jogar('melhores.txt', rede)
        elif i == 2:
            gen()
        elif i == 3:
            jogar_jogo()
