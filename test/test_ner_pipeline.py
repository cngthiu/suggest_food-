from ..inference.pipeline import Pipeline

pipe = Pipeline("serverAI/config/app.yaml")

queries = [
    "Món gà nào nhanh trong 15 phút",
    "Gợi ý món xào ít dầu với thịt heo",
    "Tôi muốn ăn món cá thanh đạm",
    "Có món canh nấu từ tôm không",
    "Cho tôi món rim từ sườn non",
    "Món bò nào phù hợp bữa trưa",
    "Tôi cần món heo ba rọi xào rau",
    "Món canh nào dễ nấu dưới 20 phút",
   "Gợi ý món ăn không dùng sữa",
    "Tôi muốn nấu món mực cay nhẹ",
    "Có món nào dùng trứng gà cho trẻ nhỏ"
]

for q in queries:
    print(f"\nQuery: {q}")
    res = pipe.nlu.extract_slots(q)
    print("Slots:", res)

    #run: 