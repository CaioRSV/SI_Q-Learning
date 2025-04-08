import numpy as np
import random
import os  # Para verificar se o arquivo existe
from connection import connect, get_state_reward

# Variável de controle se deve considerar os parâmetros de aprendizado da QTable para o treinamento do agente
ignoreQTable = False

# Ações/Estados
actions = ["left", "right", "jump"]  
num_states = 96  
num_actions = len(actions)

# Hiperparâmetros
epsilon = 1.0  # Começa alto para explorar bastante no início
epsilon_decay = 0.995  # Reduz epsilon a cada episódio
epsilon_min = 0.05  # Nunca deixa de explorar totalmente
alpha = 0.3  # Menos agressivo na atualização
gamma = 0.99  # Maior foco no futuro

targetFile = "resultado.txt"

def state_to_index(state):
    platform = int(state[:5], 2)  
    direction = int(state[5:], 2)  
    return platform * 4 + direction

# Função para carregar a Q-table
def load_q_table():
    if os.path.exists(targetFile) and not ignoreQTable:
        return np.loadtxt(targetFile)
    return np.zeros((num_states, num_actions))

# Carregar Q-table inicialmente
q_table = load_q_table()

# Conectando
socket = connect(2037)

def choose_action(state_index):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(num_actions))  
    else:
        return np.argmax(q_table[state_index])  

def train(episodes=5000):
    global epsilon, q_table  
    for episode in range(episodes):
        state, _ = get_state_reward(socket, "jump")
        state_index = state_to_index(state)
        done = False
        total_reward = 0  

        while not done:
            action_index = choose_action(state_index)
            action = actions[action_index]

            new_state, reward = get_state_reward(socket, action)
            new_state_index = state_to_index(new_state)

            # Detect jumping off a platform into a non-platform state
            was_on_platform = int(state[:5], 2) > 0  
            is_now_on_platform = int(new_state[:5], 2) > 0  

            if action == "jump" and was_on_platform and not is_now_on_platform:
                reward -= 5  # Apply penalty for jumping off a platform to nowhere

            # Atualizar Q-table
            best_next_action = np.max(q_table[new_state_index])
            q_table[state_index, action_index] += alpha * (reward + gamma * best_next_action - q_table[state_index, action_index])

            state_index = new_state_index
            total_reward += reward  

            if reward > -10:  
                done = True

        # Reduz epsilon gradualmente
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay  

        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

        # Salvar e recarregar a QTable a cada 10 episodes
        if (episode + 1) % 10 == 0:
            np.savetxt(targetFile, q_table, fmt="%.6f")
            q_table = load_q_table()  # Recarregar Q-table
            print("|---[Tabela atualizada e recarregada]---|")

    print("Treinamento concluído. Q-table salva!")


if __name__ == "__main__":
    train(100_000)  