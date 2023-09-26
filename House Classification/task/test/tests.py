from hstest.stage_test import List
from hstest import *
import ast

correct_answer = [914, 7, False, 9, 93.19, 239]
err = 0.01


class LoadTest(StageTest):

    def generate(self) -> List[TestCase]:
        return [TestCase(time_limit=1000000)]

    def check(self, reply: str, attach):

        reply = reply.strip()

        if len(reply) == 0:
            return CheckResult.wrong("No output was printed")

        if len(reply.split('\n')) != 6:
            return CheckResult.wrong(
                'The number of answers supplied does not equal 6. Make sure to put each answer on a separate line.')

        user_answer = reply.split('\n')
        try:
            reply1 = ast.literal_eval(user_answer[0])
            reply2 = ast.literal_eval(user_answer[1])
            reply3 = ast.literal_eval(user_answer[2])
            reply4 = ast.literal_eval(user_answer[3])
            reply5 = ast.literal_eval(user_answer[4])
            reply6 = ast.literal_eval(user_answer[5])
        except Exception as e:
            return CheckResult.wrong(f"Seems that output is in wrong format.\n"
                                     f"Make sure you use only the following Python structures in the output: string, int, float, list, dictionary")

        if not isinstance(reply1, int):
            return CheckResult.wrong(f'Print 1st answer as a number')

        if reply1 != correct_answer[0]:
            return CheckResult.wrong(f"Seems like your 1st  answer is not correct")

        if not isinstance(reply2, int):
            return CheckResult.wrong(f'Print 2nd answer as a number')

        if reply2 != correct_answer[1]:
            return CheckResult.wrong(f"Seems like your 2nd answer is not correct")

        if not isinstance(reply3, bool):
            return CheckResult.wrong(f'Print 3rd answer as a bool')

        if reply3 != correct_answer[2]:
            return CheckResult.wrong(f"Seems like your 3rd answer is not correct")

        if not isinstance(reply4, int):
            return CheckResult.wrong(f'Print 4th answer as a int')

        if reply4 != correct_answer[3]:
            return CheckResult.wrong(f"Seems like your 4th answer is not correct")

        if not isinstance(reply5, float):
            return CheckResult.wrong(f'Print 5th answer as a float')

        if reply5 > correct_answer[4] + err * correct_answer[4] or reply5 < correct_answer[4] - err * correct_answer[4]:
            return CheckResult.wrong(f"Seems like your 5th answer is not correct")

        if not isinstance(reply6, int):
            return CheckResult.wrong(f'Print 6th answer as a int')

        if reply6 != correct_answer[5]:
            return CheckResult.wrong(f"Seems like your 6th answer is not correct")

        return CheckResult.correct()


if __name__ == '__main__':
    LoadTest().run_tests()
