import numpy as np


class RedeNeural:
    def __init__(self, entrada, ocultos, saida):
        # Ajuste da topologia
        self.pesos_entrada_oculta = np.random.uniform(-1, 1, (entrada, ocultos))
        self.pesos_oculta_saida = np.random.uniform(-1, 1, (ocultos, saida))
        self.bias_oculta = np.random.uniform(-1, 1, ocultos)
        self.bias_saida = np.random.uniform(-1, 1, saida)

    def forward(self, x):
        self.ativacao_oculta = np.tanh(np.dot(x, self.pesos_entrada_oculta) + self.bias_oculta)
        self.saida = np.tanh(np.dot(self.ativacao_oculta, self.pesos_oculta_saida) + self.bias_saida)
        return self.saida

    def set_pesos(self, cromossomo):
        split1 = self.pesos_entrada_oculta.size
        split2 = split1 + self.pesos_oculta_saida.size
        split3 = split2 + self.bias_oculta.size

        self.pesos_entrada_oculta = cromossomo[:split1].reshape(self.pesos_entrada_oculta.shape)
        self.pesos_oculta_saida = cromossomo[split1:split2].reshape(self.pesos_oculta_saida.shape)
        self.bias_oculta = cromossomo[split2:split3]
        self.bias_saida = cromossomo[split3:]
