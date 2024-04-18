import ast
from hstest.stage_test import List
from hstest import *
import re

correct_answer_onehot = 0.64
correct_answer_ordinal = 0.86
correct_answer_te = 0.75
regex4float = "[+-]?([0-9]*[.])?[0-9]+"


class CompareTest(StageTest):

    def generate(self) -> List[TestCase]:
        return [TestCase(time_limit=1000000)]

    def check(self, reply: str, attach):

        reply = reply.strip().replace(" ", "").lower()

        if len(reply) == 0:
            return CheckResult.wrong("No output was printed")

        if len(reply.split('\n')) != 3:
            return CheckResult.wrong('The number of lines is the output does not equal 3.\n'
                                     'Please follow the format specified in the Examples section.')

        answer_onehot = re.search(pattern=f"onehotencoder:{regex4float}", string=reply)

        if answer_onehot is None:
            raise WrongAnswer(
                "Didn't find all or a part of the answer for the OneHotEncoder case.\n"
                "Check whether the format is correct")
        user_f1_onehot = float(answer_onehot.group(0).split(':')[1])

        if user_f1_onehot > correct_answer_onehot + 0.01 * correct_answer_onehot or \
                user_f1_onehot < correct_answer_onehot - 0.01 * correct_answer_onehot:
            return CheckResult.wrong(f'Seems like your answer for OneHotEncoder case is not correct.')

        answer_ordinal = re.search(pattern=f"ordinalencoder:{regex4float}", string=reply)

        if answer_ordinal is None:
            raise WrongAnswer(
                "Didn't find all or a part of the answer for the OrdinalEncoder case.\n"
                "Check whether the format is correct")
        user_f1_ordinal = float(answer_ordinal.group(0).split(':')[1])

        if user_f1_ordinal > correct_answer_ordinal + 0.01 * correct_answer_ordinal or \
                user_f1_ordinal < correct_answer_ordinal - 0.01 * correct_answer_ordinal:
            return CheckResult.wrong(f'Seems like your answer for OrdinalEncoder case is not correct.')

        answer_te = re.search(pattern=f"targetencoder:{regex4float}", string=reply)

        if answer_te is None:
            raise WrongAnswer(
                "Didn't find all or a part of the answer for the TargetEncoder case.\n"
                "Check whether the format is correct")
        user_f1_te = float(answer_te.group(0).split(':')[1])

        if user_f1_te > correct_answer_te + 0.01 * correct_answer_te or \
                user_f1_te < correct_answer_te - 0.01 * correct_answer_te:
            return CheckResult.wrong(f'Seems like your answer for TargetEncoder case is not correct.')

        return CheckResult.correct()


if __name__ == '__main__':
    CompareTest().run_tests()