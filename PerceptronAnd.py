import pandas as pd

weight1 = 1.0
weight2 = 1.0
bias = -1.5

test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
correct_outputs = [False, False, False, True]
outputs = []

for testInput, correctOutput in zip(test_inputs, correct_outputs):
    linearCombination = weight1 * \
        testInput[0] + weight2 * testInput[1] + bias
    output = int(linearCombination >= 0)
    isCorrectStr = 'Yes' if output == correctOutput else 'No'
    outputs.append([testInput[0], testInput[1],
                    linearCombination, output, isCorrectStr])

num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
output_frame = pd.DataFrame(outputs, columns=[
    'Input 1', 'Input 2', ' Linear Combination', 'Ouput', 'Is Correct'])
if not num_wrong:
    print('Nice! You got it all correct.\n')
else:
    print('You got {} wrong. Keep trying!\n'.format(num_wrong))

print(output_frame.to_string())
