import json
import os
# from pprint import pprint
# from report_model.output_report import OutputReport
import pandas as pd


class SummaryReport:
    def __init__(self):
        pass
        # self.generate_summary_report(run_id)

    # @classmethod
    # def from_dict(cls, dict):
    #     obj = cls()
    #     obj.__dict__.update(dict)
    #     return obj

    def generate_summary_report(self, run_id):

        output_report_list = []
        path = 'D:\private-projects\\linear-dim-reduction\\report_jsons'
        for filename in os.listdir(path):
            with open(path + '\\' + filename) as f:
                output_report = json.load(f)
            if output_report["run_id"] == run_id:
                output_report_list.append(output_report)
            #print(filename)
        #print(output_report_list)

        summary_report_df = pd.DataFrame(output_report_list)
        # for output_r in output_report_list:
        #     one_report_df = pd.DataFrame.from_dict(output_r, orient='index')
        #     summary_report_df.append(one_report_df)

        pass
        return summary_report_df


        #output_report: OutputReport
        # with open('D:\\private-projects\\linear-dim-reduction\\report_jsons\\2018-11-03 141735 output_report.json') as f:
        #     output_report = json.load(f)
        # #x = json.loads(data, object_hook=SummaryReport.from_dict)
        # #pprint(x)
        # pprint(output_report)

    def make_report_pretty(self, summary_report_df):

        pass

    def gen_docx_from_df(self, df):
        import docx
        #import pandas as pd

        # i am not sure how you are getting your data, but you said it is a
        # pandas data frame
        #df = pd.DataFrame(data)

        # open an existing document
        doc = docx.Document('D:\\private-projects\\linear-dim-reduction\\summary_report\\test2.docx')

        # add a table to the end and create a reference variable
        # extra row is so we can add the header row
        t = doc.add_table(df.shape[0]+1, df.shape[1])

        # add the header rows.
        for j in range(df.shape[-1]):
            t.cell(0,j).text = df.columns[j]

        # add the rest of the data frame
        for i in range(df.shape[0]):
            for j in range(df.shape[-1]):
                t.cell(i+1,j).text = str(df.values[i,j])

        doc.save('D:\\private-projects\\linear-dim-reduction\\summary_report\\test3.docx')


    def open_json_file(self, path):
        with open('D:\\private-projects\\linear-dim-reduction\\report_jsons\\2018-11-03 141735 output_report.json') as f:
            output_report = json.load(f)


# sr = SummaryReport()
# report_df = sr.generate_summary_report(run_id='2018-11-03 141716')

sr = SummaryReport()
report_df = sr.generate_summary_report(run_id='2018-11-03 141716')
#sr.gen_docx_from_df(report_df)



