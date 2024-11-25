import random
from collections import defaultdict
from typing import Dict, Tuple, List
import math


class QLearningMDP:
    def __init__(self, initial_alpha: float = 0.2, discount_rate: float = 0.99, alpha_decay: float = 0.995):
        self.alpha = initial_alpha
        self.gamma = discount_rate
        self.alpha_decay = alpha_decay

        # Define states
        self.states = {
            'RU8p', 'TU10p', 'RU10p', 'RD10p', 'RU8a', 'RD8a',
            'TU10a', 'RU10a', 'RD10a', 'TD10a'
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

        # Initialize Q-values to 0
        self.q_values = defaultdict(float)
        for state in self.states:
            for action in self.actions[state]:
                self.q_values[(state, action)] = 0.0

        # Terminal state rewards
        self.terminal_rewards = {
            'TU10a': -1,
            'RU10a': 0,
            'RD10a': 4,
            'TD10a': 3
        }

    def get_next_state_and_reward(self, state: str, action: str) -> Tuple[str, float]:
        """Determine next state and reward based on current state and action"""
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
                # Probabilistic transition
                if random.random() < 0.5:
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

        elif state in self.terminal_rewards:
            return '11am class begins', self.terminal_rewards[state]

        return None, 0

    def get_max_q_value(self, state: str) -> float:
        """Get maximum Q-value for a state across all possible actions"""
        if state == '11am class begins':
            return 0
        return max(self.q_values[(state, action)] for action in self.actions[state])

    def run_episode(self) -> float:
        """Run a single episode and return the maximum Q-value change"""
        current_state = 'RU8p'  # Starting state
        max_change = 0

        while current_state != '11am class begins':
            # Choose random action (equiprobable policy)
            possible_actions = self.actions[current_state]
            action = random.choice(possible_actions)

            # Get current Q-value
            old_q = self.q_values[(current_state, action)]

            # Take action
            next_state, reward = self.get_next_state_and_reward(current_state, action)

            # Calculate new Q-value
            max_next_q = self.get_max_q_value(next_state)
            new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)

            # Update Q-value and track maximum change
            self.q_values[(current_state, action)] = new_q
            change = abs(new_q - old_q)
            max_change = max(max_change, change)

            # Print update details
            print(f"\nState: {current_state}, Action: {action}")
            print(f"Previous Q-value: {old_q:.3f}")
            print(f"Immediate reward: {reward}")
            print(f"Max Q-value for next state: {max_next_q:.3f}")
            print(f"New Q-value: {new_q:.3f}")

            # Move to next state
            current_state = next_state

        return max_change

    def get_optimal_policy(self) -> Dict[str, str]:
        """Determine optimal policy based on final Q-values"""
        policy = {}
        for state in self.states:
            if state in self.terminal_rewards:
                policy[state] = 'any'
            else:
                policy[state] = max(
                    self.actions[state],
                    key=lambda a: self.q_values[(state, a)]
                )
        return policy


def run_q_learning(threshold: float = 0.001):
    print("Starting Q-Learning for Student MDP...")
    mdp = QLearningMDP()
    episode = 0

    while True:
        episode += 1
        print(f"\nEpisode {episode}")
        print("-" * 50)

        # Run episode and get maximum change
        max_change = mdp.run_episode()
        print(f"\nMaximum Q-value change in episode: {max_change:.3f}")

        # Decay learning rate
        mdp.alpha *= mdp.alpha_decay
        print(f"New learning rate: {mdp.alpha:.3f}")

        if max_change < threshold:
            break

    # Print final results
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"\nConverged after {episode} episodes")

    print("\nFinal Q-values:")
    for state in sorted(mdp.states):
        print(f"\nState: {state}")
        for action in mdp.actions[state]:
            print(f"  {action}: {mdp.q_values[(state, action)]:.3f}")

    optimal_policy = mdp.get_optimal_policy()
    print("\nOptimal Policy:")
    for state in sorted(optimal_policy.keys()):
        print(f"{state}: {optimal_policy[state]}")


if __name__ == "__main__":
    run_q_learning()