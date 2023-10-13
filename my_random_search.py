from itertools import product
# TODO: add docstrings


class MyRandomSearch:

    def __init__(self, iperparameters: dict[str, list], max_iterations: int = None) -> None:
        for key, values in iperparameters.items():
            if len(values) != len(set(values)):
                raise ValueError(
                    f'Duplicate values in iperparameters dict for {key} key')

        self.input_dict = iperparameters
        self.keys = list(iperparameters.keys())
        self.value_combinations = self._generate_combinations()
        self.max_iterations = max_iterations if max_iterations is not None else len(
            self.value_combinations)

        if self.max_iterations > len(self.value_combinations):
            raise ValueError(
                f'Max iteration {self.max_iterations} given, but only {len(self.value_combinations)} possible')

    def _generate_combinations(self) -> list:
        values_lists = [self.input_dict[key] for key in self.keys]
        return list(product(*values_lists))

    def __iter__(self):
        self.current_iteration = 0
        return self

    def __next__(self) -> dict[str, any]:
        if self.current_iteration >= self.max_iterations or not self.value_combinations:
            raise StopIteration

        combination = self.value_combinations.pop()

        result_dict = {key: value for key,
                       value in zip(self.keys, combination)}

        self.current_iteration += 1
        return result_dict
