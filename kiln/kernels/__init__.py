"""NKI kernels run inside LNL graphs. A module here builds its kernel only where the nki package
imports (the Neuron venv), so the CPU path (tests, the reference model) never needs it."""
