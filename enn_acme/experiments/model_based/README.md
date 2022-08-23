# A model-based ENN ACME agent

The agent model has the following components.

- A representation module that maps observations to hidden states.
- A dynamics module that maps hidden states and actions to the next hidden states.
- A prediction module that maps hidden states to reward, value, and policy predictions.

The agent is trained on its reward, value, and policy predictions. It selects
actions according to either its action value predictions or policy predictions.
