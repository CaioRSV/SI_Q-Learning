import numpy as np

# Definições
num_states = 96
num_actions = 3  # "left", "right", "jump"

# Criar Q-table zerada
q_table = np.zeros((num_states, num_actions))

# Salvar no arquivo resultado.txt
np.savetxt("resultado.txt", q_table, fmt="%.6f")

print("Q-table resetada e salva em 'resultado.txt'.")