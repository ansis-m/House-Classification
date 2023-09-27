import ast
from hstest.stage_test import List
from hstest import *

correct_answer = {'KV': 76, 'EE': 7, 'AD': 6, 'VA': 6, 'NX': 5, 'XP': 5, 'KB': 5, 'VG': 5, 'CR': 5, 'LW': 5, 'AW': 5,
                  'AK': 4, 'EM': 4, 'SC': 4, 'KN': 4, 'AE': 4, 'LN': 4, 'GJ': 4, 'EZ': 4, 'VE': 4, 'LD': 4, 'CW': 4,
                  'KG': 4, 'XR': 4, 'CJ': 4, 'GP': 4, 'JR': 4, 'KH': 4, 'SM': 4, 'NM': 4, 'GG': 4, 'AH': 4, 'BG': 4,
                  'VS': 4, 'VX': 4, 'NB': 4, 'KJ': 4, 'HB': 4, 'JX': 4, 'KA': 4, 'GL': 4, 'KP': 4, 'CA': 3, 'WT': 3,
                  'MP': 3, 'LG': 3, 'TH': 3, 'RD': 3, 'TS': 3, 'GE': 3, 'AP': 3, 'EX': 3, 'CK': 3, 'ZW': 3, 'HN': 3,
                  'SX': 3, 'PJ': 3, 'RW': 3, 'KM': 3, 'BD': 3, 'JK': 3, 'WX': 3, 'XJ': 3, 'EH': 3, 'LM': 3, 'HJ': 3,
                  'GD': 3, 'XW': 3, 'EW': 3, 'GW': 3, 'MT': 3, 'DE': 3, 'DS': 3, 'ER': 3, 'RZ': 3, 'JN': 3, 'TT': 3,
                  'DJ': 3, 'CN': 3, 'BE': 3, 'TV': 3, 'VM': 3, 'JC': 3, 'KE': 3, 'RK': 3, 'BJ': 3, 'NK': 3, 'AN': 3,
                  'ZM': 3, 'HA': 2, 'ST': 2, 'NT': 2, 'VB': 2, 'EC': 2, 'EB': 2, 'XX': 2, 'VL': 2, 'VP': 2, 'GX': 2,
                  'CP': 2, 'AM': 2, 'VD': 2, 'TL': 2, 'NJ': 2, 'BL': 2, 'TM': 2, 'WV': 2, 'KD': 2, 'GB': 2, 'DM': 2,
                  'CS': 2, 'PZ': 2, 'WJ': 2, 'PA': 2, 'NZ': 2, 'VC': 2, 'PT': 2, 'XE': 2, 'BT': 2, 'CM': 2, 'BR': 2,
                  'ND': 2, 'EA': 2, 'EK': 2, 'RT': 2, 'XT': 2, 'MD': 2, 'ML': 2, 'KL': 2, 'LX': 2, 'ET': 2, 'AV': 2,
                  'BM': 2, 'EP': 2, 'RN': 2, 'HW': 2, 'GH': 2, 'BA': 2, 'PE': 2, 'XL': 2, 'DG': 2, 'BH': 2, 'ZC': 2,
                  'DB': 2, 'BS': 2, 'GN': 2, 'JS': 2, 'TE': 2, 'XH': 2, 'DP': 2, 'DH': 2, 'DT': 2, 'NV': 2, 'MA': 2,
                  'EL': 2, 'BW': 2, 'GA': 2, 'ES': 2, 'RX': 2, 'LL': 2, 'HH': 2, 'AG': 2, 'WG': 2, 'AR': 2, 'VJ': 2,
                  'HR': 2, 'RA': 2, 'RS': 2, 'HD': 2, 'ZP': 2, 'AA': 2, 'TP': 2, 'NH': 2, 'AL': 2, 'PW': 2, 'DZ': 2,
                  'ED': 2, 'GZ': 2, 'WE': 2, 'LH': 2, 'GV': 2, 'MR': 2, 'RL': 2, 'VK': 1, 'BZ': 1, 'TZ': 1, 'KW': 1,
                  'AX': 1, 'HP': 1, 'LS': 1, 'JE': 1, 'DC': 1, 'XN': 1, 'PX': 1, 'KS': 1, 'VH': 1, 'ZV': 1, 'RC': 1,
                  'MN': 1, 'RH': 1, 'BP': 1, 'MB': 1, 'PM': 1, 'CL': 1, 'LJ': 1, 'WR': 1, 'CB': 1, 'HE': 1, 'XB': 1,
                  'PG': 1, 'PN': 1, 'NR': 1, 'BK': 1, 'DK': 1, 'JG': 1, 'CG': 1, 'TA': 1, 'BC': 1, 'TR': 1, 'DN': 1,
                  'TD': 1, 'DA': 1, 'HX': 1, 'WH': 1, 'AZ': 1, 'KR': 1, 'JW': 1, 'KK': 1, 'TJ': 1, 'LK': 1, 'MZ': 1,
                  'KC': 1, 'XA': 1, 'RV': 1, 'BN': 1, 'PH': 1, 'VR': 1, 'AT': 1, 'SH': 1}

class SplitTest(StageTest):

    def generate(self) -> List[TestCase]:
        return [TestCase(time_limit=1000000)]

    def check(self, reply: str, attach):

        reply = reply.strip()

        if len(reply) == 0:
            return CheckResult.wrong("No output was printed")

        if reply.count('{') < 1 or reply.count('}') < 1:
            return CheckResult.wrong('Print the answer as a dictionary')

        if len(reply.split('\n')) != 1:
            return CheckResult.wrong('The number of answers supplied does not equal 1. Provide one Python dictionary as specified in the stage description.')

        index_from = reply.find('{')
        index_to = reply.rfind('}')
        dict_str = reply[index_from: index_to + 1]
        try:
            user_dict = ast.literal_eval(dict_str)
        except Exception as e:
            return CheckResult.wrong(f"Seems that output is in wrong format.\n"
                                     f"Make sure you use only the following Python structures in the output: string, int, float, list, dictionary.")

        if not isinstance(user_dict, dict):
            return CheckResult.wrong(f'Print answer as a dictionary')

        if len(user_dict.keys()) != len(correct_answer.keys()):
            return CheckResult.wrong(
                f'Output should be a dictionary with {len(correct_answer.keys())} elements,'
                f'\nfound {len(user_dict.keys())} elements')

        for key in correct_answer.keys():
            if key not in user_dict.keys():
                return CheckResult.wrong(f'Output should contain {key} as key')

        for key in user_dict.keys():
            if key not in correct_answer.keys():
                return CheckResult.wrong(
                    f'Output should not contain {key} as key.')
            if key in correct_answer.keys():
                if user_dict[key] != correct_answer[key]:
                    return CheckResult.wrong(f"Seems like answer is not correct;\n"
                                             f"Check element with a key {key} of your dictionary")

        return CheckResult.correct()


if __name__ == '__main__':
    SplitTest().run_tests()