from typing import Dict, Tuple, List
import math


class ValueIterationMDP:
    def __init__(self, discount_rate: float = 0.99):
        self.gamma = discount_rate

        self.states = [
            'RU8p', 'TU10p', 'RU10p', 'RD10p', 'RU8a', 'RD8a',
            'TU10a', 'RU10a', 'RD10a', 'TD10a'
        ]

        self.values = {state: 0.0 for state in self.states}

        self.actions = {
            'RU8p': ['P', 'R', 'S'],
            'TU10p': ['P', 'R'],
            'RU10p': ['P', 'R', 'S'],
            'RD10p': ['R', 'P'],
            'RU8a': ['P', 'R', 'S'],
            'RD8a': ['R', 'P'],
            'TU10a': ['any'],  # Terminal state
            'RU10a': ['any'],  # Terminal state
            'RD10a': ['any'],  # Terminal state
            'TD10a': ['any']  # Terminal state
        }
        # Terminal state rewards
        self.terminal_rewards = {
            'TU10a': -1,
            'RU10a': 0,
            'RD10a': 4,
            'TD10a': 3
        }

    def get_action_value(self, state: str, action: str) -> float:
        """Calculate the value of taking an action in a state"""
        if state in self.terminal_rewards:
            return self.terminal_rewards[state]

        if state == 'RU8p':
            if action == 'P':
                return 2 + self.gamma * self.values['TU10p']
            elif action == 'R':
                return 0 + self.gamma * self.values['RU10p']
            else:  # S
                return -1 + self.gamma * self.values['RD10p']

        elif state == 'TU10p':
            if action == 'P':
                return 2 + self.gamma * self.values['TU10a']
            else:  # R
                return 0 + self.gamma * self.values['RU8a']

        elif state == 'RU10p':
            if action == 'P':
                # Probabilistic transition
                return 2 + self.gamma * (0.5 * self.values['RU8a'] + 0.5 * self.values['TU10a'])
            elif action == 'R':
                return 0 + self.gamma * self.values['RU8a']
            else:  # S
                return -1 + self.gamma * self.values['RD10a']

        elif state == 'RD10p':
            if action == 'R':
                return 0 + self.gamma * self.values['RD8a']
            else:  # P
                return 2 + self.gamma * self.values['TD10a']

        elif state == 'RU8a':
            if action == 'P':
                return 2 + self.gamma * self.values['TU10a']
            elif action == 'R':
                return 0 + self.gamma * self.values['RU10a']
            else:  # S
                return -1 + self.gamma * self.values['RD10a']

        elif state == 'RD8a':
            if action == 'R':
                return 0 + self.gamma * self.values['RD10a']
            else:  # P
                return 2 + self.gamma * self.values['TD10a']

        return 0

    def update_state(self, state: str) -> Tuple[float, str, Dict[str, float]]:
        """Update value for a single state and return (max_value, best_action, action_values)"""
        if state in self.terminal_rewards:
            return self.terminal_rewards[state], 'any', {'any': self.terminal_rewards[state]}

        action_values = {
            action: self.get_action_value(state, action)
            for action in self.actions[state]
        }

        best_action = max(action_values.items(), key=lambda x: x[1])[0]
        max_value = action_values[best_action]

        return max_value, best_action, action_values

    def value_iteration(self, threshold: float = 0.001) -> Tuple[Dict[str, str], int]:
        """Run value iteration until convergence"""
        iteration = 0
        while True:
            iteration += 1
            max_delta = 0

            print(f"\nIteration {iteration}")
            print("-" * 50)

            # Update each state
            for state in self.states:
                old_value = self.values[state]
                new_value, best_action, action_values = self.update_state(state)

                # Print detailed update information
                print(f"\nState: {state}")
                print(f"Previous value: {old_value:.3f}")
                print("Action values:")
                for action, value in action_values.items():
                    print(f"  {action}: {value:.3f}")
                print(f"Best action: {best_action}")
                print(f"New value: {new_value:.3f}")

                self.values[state] = new_value
                max_delta = max(max_delta, abs(new_value - old_value))

            print(f"\nMaximum change in this iteration: {max_delta:.3f}")

            if max_delta < threshold:
                break

        # Compute final policy
        policy = {}
        for state in self.states:
            if state in self.terminal_rewards:
                policy[state] = 'any'
            else:
                _, best_action, _ = self.update_state(state)
                policy[state] = best_action

        return policy, iteration


def run_value_iteration():
    print("Starting Value Iteration for Student MDP...")
    mdp = ValueIterationMDP()

    # Run value iteration
    optimal_policy, num_iterations = mdp.value_iteration()

    # Print final results
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"\nConverged after {num_iterations} iterations\n")

    print("Final State Values:")
    for state in sorted(mdp.states):
        print(f"{state}: {mdp.values[state]:.3f}")

    print("\nOptimal Policy:")
    for state in sorted(optimal_policy.keys()):
        print(f"{state}: {optimal_policy[state]}")


if __name__ == "__main__":
    run_value_iteration()