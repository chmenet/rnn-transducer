from unittest import TestCase
from rnnt.utils_aihub import computer_cer


class TestComputerCer(TestCase):

    def setUp(self) -> None:
        pass

    def test_computer_cer(self):
        """
        predictions 와 ground truths의 거리를 반환받기
        :return:
        """
        test_data = {
            'preds': [
                [0, 1, 1],
                [1, 1]
            ],
            'labels': [
                [0, 1, 1],
                [1, 1]
            ]
        }

        expected_dist = 0
        expected_total = 5

        real_dist, real_total = computer_cer(test_data['preds'], test_data['labels'])
        print(real_dist, real_total)
        self.assertTrue((expected_dist==real_dist) and (expected_total == real_total))

    def test_compute_cer_with_diff_predictions(self):

        test_data = {
            'preds': [
                [c for c in '오는 날시가어때요']
            ],
            'labels': [
                [c for c in '오늘은날씨가어때']
            ]
        }

        """
        substitution: 2개
        deletion: 1개
        insertion: 1개
        length of ref: 8개 ('오늘은날씨가어때')
        """

        expected_dist = 4
        expected_total = 8

        real_dist, real_total = computer_cer(test_data['preds'], test_data['labels'])
        print(real_dist, real_total)
        self.assertTrue((expected_dist == real_dist) and (expected_total == real_total))

    def tearDown(self) -> None:
        pass
