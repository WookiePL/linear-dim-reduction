import json


class OutputReport:
    def __init__(self,
                 run_id,
                 input_params,
                 plot_training_png_url,
                 plot_test_png_url,
                 f1_score,
                 error,
                 accuracy,
                 precision,
                 recall,
                 conf_matrix,
                 conf_matrix_png_url):
        self.run_id = run_id
        self.input_params = input_params
        self.plot_training_png_url = plot_training_png_url
        self.plot_test_png_url = plot_test_png_url
        self.f1_score = f1_score
        self.error = error
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.conf_matrix_png_url = conf_matrix_png_url
        self.conf_matrix = conf_matrix

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=False, indent=4)
