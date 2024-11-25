import random
from collections import defaultdict
from typing import List, Dict, Tuple


class StudentMDP:
    def __init__(self):
        # Define states
        self.states = {
            'RU8p', 'TU10p', 'RU10p', 'RD10p', 'RU8a', 'RD8a', 'TU10a', 'RU10a', 'RD10a', 'TD10a'
        }

        # Define possible actions for each state
        self.actions = {
            'RU8p': ['P', 'R', 'S'],
            'TU10p': ['P', 'R'],
            'RU10p': ['P', 'R', 'S'],
            'RD10p': ['R', 'P'],
            'RU8a': ['P', 'R', 'S'],
            'RD8a': ['R', 'P'],
            'TU10a': ['any'],
            'RU10a': ['any'],
            'RD10a': ['any'],
            'TD10a': ['any']
        }

        # Initialize state values
        self.state_values = {state: 0.0 for state in self.states}
        self.state_visits = defaultdict(int)

    def get_next_state_and_reward(self, state: str, action: str) -> Tuple[str, float]:
        if state == 'RU8p':
            if action == 'P':
                return 'TU10p', 2
            elif action == 'R':
                return 'RU10p', 0
            else:  # S
                return 'RD10p', -1

        elif state == 'TU10p':
            if action == 'P':
                return 'TU10a', 2
            else:  # R
                return 'RU8a', 0

        elif state == 'RU10p':
            if action == 'P':
                prob = random.random()
                if prob < 0.5:
                    return 'RU8a', 2
                else:
                    return 'TU10a', 2
            elif action == 'R':
                return 'RU8a', 0
            else:  # S
                return 'RD10a', -1

        elif state == 'RD10p':
            if action == 'R':
                return 'RD8a', 0
            else:  # P
                return 'TD10a', 2

        elif state == 'RU8a':
            if action == 'P':
                return 'TU10a', 2
            elif action == 'R':
                return 'RU10a', 0
            else:  # S
                return 'RD10a', -1

        elif state == 'RD8a':
            if action == 'R':
                return 'RD10a', 0
            else:  # P
                return 'TD10a', 2

        elif state in ['TU10a', 'RU10a', 'RD10a', 'TD10a']:  # Terminal states
            return '11am class begins', {'TU10a': -1, 'RU10a': 0, 'RD10a': 4, 'TD10a': 3}[state]

        return None, 0

    def run_episode(self) -> List[Tuple[str, str, float]]:
        episode = []
        current_state = 'RU8p'  # Starting state

        while current_state != '11am class begins':
            # Choose random action
            possible_actions = self.actions[current_state]
            action = random.choice(possible_actions)

            # Get next state and reward
            next_state, reward = self.get_next_state_and_reward(current_state, action)

            # Record experience
            episode.append((current_state, action, reward))

            # Move to next state
            current_state = next_state

        return episode

    def update_state_values(self, episode: List[Tuple[str, str, float]], alpha: float = 0.1):
        # Calculate returns for each state
        G = 0
        visited_states = set()

        for state, _, reward in reversed(episode):
            G += reward

            if state not in visited_states:  # First-visit MC
                visited_states.add(state)
                self.state_visits[state] += 1
                # Update state value using running average
                self.state_values[state] += alpha * (G - self.state_values[state])


def run_monte_carlo_simulation(num_episodes: int = 50):
    mdp = StudentMDP()
    total_rewards = []

    print("Starting Monte Carlo simulation for Student MDP...")
    print("\nEpisode Experiences:")

    for episode_num in range(num_episodes):
        episode = mdp.run_episode()
        episode_reward = sum(reward for _, _, reward in episode)
        total_rewards.append(episode_reward)

        # Print episode experience
        print(f"\nEpisode {episode_num + 1}:")
        print("State -> Action -> Reward")
        for state, action, reward in episode:
            print(f"{state} -> {action} -> {reward}")
        print(f"Total episode reward: {episode_reward}")

        # Update state values
        mdp.update_state_values(episode)

    # Print final results
    print("\nFinal State Values:")
    for state in sorted(mdp.state_values.keys()):
        print(f"{state}: {mdp.state_values[state]:.2f}")

    # Print episode rewards statistics
    print("\nEpisode Rewards Summary:")
    print(f"Average reward per episode: {sum(total_rewards) / len(total_rewards):.2f}")
    print(f"Minimum reward: {min(total_rewards):.2f}")
    print(f"Maximum reward: {max(total_rewards):.2f}")


if __name__ == "__main__":
    run_monte_carlo_simulation()