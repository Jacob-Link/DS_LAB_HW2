import pandas as pd
from pathlib import Path
from model_loader import get_missed_labels
import numpy as np
import xlsxwriter
from torchvision.utils import save_image


def extract_data_from_missed(missed):
    res_list = []
    for k, v in missed.items():
        if type(v) == type(dict()):
            if type(v["losses"]) == type(np.array([])):
                for i in range(len(v["losses"])):
                    tempt_dict = {"img": v["img"][i], "gt": v["gt"][i].item(), "pred": v["pred"][i].item(),
                                  "loss": round(v["losses"][i][0], 5)}
                    res_list.append(tempt_dict)
            else:
                tempt_dict = {"img": v["img"][0], "gt": v["gt"][0], "pred": v["pred"][0], "loss": v["losses"]}
                res_list.append(tempt_dict)
    return res_list


def create_excel_sheet_with_error_info(res_list, title):
    # example to saving image save_image(res_list[0]["img"][0], file_path_to_save/"file_name.png")

    # test_1_row
    rows = res_list
    rows.sort(key=lambda x: x["loss"], reverse=True)

    workbook = xlsxwriter.Workbook(f"{title}.xlsx")
    worksheet = workbook.add_worksheet()

    # Widen the first column to make the text clearer.
    worksheet.set_column("A:A", 10)

    cell_format_title = workbook.add_format()
    cell_format_title.set_bold(True)
    cell_format_title.set_center_across()

    # define the format we will use during the writing stage
    cell_format = workbook.add_format()
    cell_format.set_center_across(True)

    worksheet.write("A1", "img", cell_format_title)
    worksheet.write("B1", "true label", cell_format_title)
    worksheet.write("C1", "predicted", cell_format_title)
    worksheet.write("D1", "loss", cell_format_title)

    # write data to cells
    for i in range(len(rows)):
        save_image(rows[i]["img"][0], Path.cwd() / "error_imgs" / f"{i}.png")
        worksheet.insert_image(f"A{i * 4 + 2}", Path.cwd() / "error_imgs" / f"{i}.png", {"x_offset": 5, "y_offset": 5})
        worksheet.write(f"B{i * 4 + 3}", rows[i]["gt"], cell_format)
        worksheet.write(f"C{i * 4 + 3}", rows[i]["pred"], cell_format)
        worksheet.write(f"D{i * 4 + 3}", rows[i]["loss"], cell_format)

    workbook.close()
    print(f">>> successfully exported error analysis info to excel. {len(rows):,} images")


def print_distribution_of_errors_by_class(res_list, class_list, export=False):
    total_labels = len(res_list)
    class_counter = {x: 0 for x in class_list}

    for x in res_list:
        class_counter[x["gt"]] += 1

    data = []
    for key, val in class_counter.items():
        tup = (key, val, f"{round(100 * val / total_labels, 2)}%")
        data.append(tup)
    data.sort(key=lambda x: x[1], reverse=True)

    print_df = pd.DataFrame(data, columns=["class", "count", "%"])
    print("----------  label balance ----------")
    print(print_df)
    print(f"Size of miss labelled data: {sum(print_df['count']):,}")
    print("------------------------------------")
    print()
    if export:
        print_df.to_csv("error_analysis_class_distribution.csv", index=False)


def update_labels_to_numerals(res_list, mapping):
    for dict in res_list:
        try:
            dict["gt"] = mapping[dict["gt"]]
        except KeyError:
            dict["gt"] = mapping[dict["gt"].item()]
        try:
            dict["pred"] = mapping[dict["pred"]]
        except KeyError:
            dict["pred"] = mapping[dict["pred"].item()]
    return res_list


if __name__ == '__main__':
    val_data_path = Path.cwd().parent / 'error_analysis' / 'attempt_4' / 'val'
    model_path = Path.cwd().parent / 'models' / "attempt_4_model.pt"

    missed, index_to_class_map = get_missed_labels(val_data_path, model_path)
    res_list = extract_data_from_missed(missed)
    update_labels_to_numerals(res_list, index_to_class_map)

    export_excel = True
    print_dist = True
    if print_dist:
        print_distribution_of_errors_by_class(res_list, list(index_to_class_map.values()), True)
    if export_excel:
        create_excel_sheet_with_error_info(res_list, "attempt_4_val_error_analysis")
