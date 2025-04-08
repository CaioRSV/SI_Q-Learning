import numpy as np
import random
import os
from connection import connect, get_state_reward

# Variável que define se o agente vai estar aprendendo ou reproduzindo o aprendizado da QTable com os valores atuais dela
onlyPlay = True
# Variável de controle se deve considerar os parâmetros de aprendizado da QTable para o treinamento do agente
ignoreQTable = False

# Ações/Estados
actions = ["jump", "left", "right"]
num_states = 96
num_actions = len(actions)

# Hiperparâmetros
epsilon = 1.0            # Exploração alta no início (epsilon-greedy)
epsilon_decay = 0.995    # Redução gradual de epsilon
epsilon_min = 0.05       # Exploração mínima
alpha_base = 0.25        # Valor de aprendizado alpha
alpha_min = 0.05         # Valor mínimo para alfa após decaimento
gamma = 0.9              # Fator de desconto foresight

targetFile = "resultado.txt"

# Cria index único com base na plataforma e direção extraídos do state
def state_to_index(state):
    platform = int(state[2:7], 2) # Convertendo plataforma de binário para inteiro
    direction = int(state[7:9], 2) # Convertendo direção de binário para inteiro
    return platform * 4 + (direction % 4) # Combinação da plataforma atual e a direção

# Carrega Q-Table se estiver disponível (e não for setado para ignorar ela no início)
def load_q_table():
    if os.path.exists(targetFile) and not ignoreQTable:
        return np.loadtxt(targetFile)
    return np.zeros((num_states, num_actions))

# Carregar Q-Table
q_table = load_q_table()

# Conexão na port do jogo
socket = connect(2037)

def choose_action(state_index):
    # Abordagem Epsilon-Greedy pra escolher a próxima ação
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, num_actions - 1) # Ação aleatória
    else:
        return int(np.argmax(q_table[state_index])) # A de maior valor Q para o estado

def q_update(value, alpha, gamma, reward, max_next):
    # Baseado na fórmula de update de Q-learning
    # Q(s, a) = (1 - alpha) * Q(s, a) + alpha * (reward + gamma * max_next)
    return (1 - alpha) * value + alpha * (reward + gamma * max_next)

def train(episodes=5000):
    global epsilon, q_table
    for episode in range(episodes):
        # Inicia o episódio com a ação "jump"
        state, _ = get_state_reward(socket, "jump")
        state_index = state_to_index(state)
        total_reward = 0
        done = False

        # Atualiza alfa de forma dinâmica por episódio:
        current_alpha = max(alpha_min, alpha_base * (epsilon_decay ** episode))

        while not done:
            action_index = choose_action(state_index)
            action = actions[action_index]

            new_state, reward = get_state_reward(socket, action)
            new_state_index = state_to_index(new_state)

            # Printa detalhes do estado e recompensa
            print(f"Estado: {new_state} | Recompensa: {reward}")
            platform = int(new_state[2:7], 2)
            direction = int(new_state[7:9], 2)
            print(f"Plataforma: {platform} | Direcao: {direction}")

            # Atualiza a Q-table usando o valor máximo do próximo estado
            max_next = np.max(q_table[new_state_index])
            current_q = q_update(q_table[state_index, action_index], current_alpha, gamma, reward, max_next)
            q_table[state_index, action_index] = current_q

            # Prepara para o próximo passo
            state = new_state
            state_index = new_state_index
            total_reward += reward

            # Condição de término do episódio (chegou a ficar melhor que -10)
            if reward > -10:
                done = True

        # Decair epsilon gradualmente
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Episódio {episode + 1}/{episodes}, Reward Total: {total_reward}, Epsilon: {epsilon:.4f}, Alpha: {current_alpha:.4f}")

        # Salva e recarrega a Q-table a cada 10 episódios
        if (episode + 1) % 10 == 0:
            np.savetxt(targetFile, q_table, fmt="%.6f")
            q_table = load_q_table()  # Recarrega a Q-table
            print("|---[Tabela atualizada e recarregada]---|")

    print("Treinamento concluído. Q-table salva.")

def play(steps=100):
    global epsilon
    epsilon = 0.0  # Greedy total, só pra ilustração

    state, _ = get_state_reward(socket, "jump")
    state_index = state_to_index(state)

    for _ in range(steps):
        action_index = choose_action(state_index)
        action = actions[action_index]

        new_state, reward = get_state_reward(socket, action)
        new_state_index = state_to_index(new_state)

        print(f"Ação: {action}, Estado: {new_state}, Recompensa: {reward}")

        state_index = new_state_index

if __name__ == "__main__":
    if (onlyPlay):
        play(800)
    else:
        train(50_000)
