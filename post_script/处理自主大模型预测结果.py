import codecs
from collections import defaultdict
import random
import json
input_file = "/Users/mingsong/Downloads/predict_result_online.json"
output_file = "online_service_glm.txt"
f_save = open(output_file, "w", encoding="utf-8")
with open(input_file,"r",encoding="utf-8") as f:
    for line in f.readlines():
        res_dict = json.loads(line.strip("\n"))
        input_data = res_dict["input"].split("\n")
        predict_data = res_dict["predict_after"].split("\n")

        app = ""
        gen_title = ""
        gen_desc = ""

        # 找到app
        for item in input_data:
            if item.startswith("app："):
                app = item.strip("app：").strip()

        # 找到生成标题和描述
        for item in predict_data:
            if item.startswith("搜索广告标题："):
                gen_title = item.strip("搜索广告标题：").strip()
            elif item.startswith("标题：") and gen_title=="":
                gen_title = item.strip("标题：").strip()
            elif item.startswith("描述："):
                gen_desc = item.strip("描述：").strip()
            elif item.startswith("搜索广告描述：") and gen_desc=="":
                gen_desc = item.strip("搜索广告描述：").strip()

        if app =="" or gen_title =="" or gen_desc=="":
            continue

        f_save.write("\t".join([app, gen_title,gen_desc])+"\n")