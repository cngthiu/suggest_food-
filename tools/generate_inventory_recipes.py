import json
import os

# Đường dẫn lưu file
OUTPUT_DIR = "serverAI/data/recipes"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Danh sách 30 món ăn từ nguyên liệu có sẵn trong kho
recipes = [
    # --- NHÓM THỊT HEO (5 món) ---
    {
        "id": "suon-non-kho-thom-30p",
        "title": "Suon non kho thom 30 phut",
        "summary": "Suon non kho voi thom (dua) chua ngot, dam da dua com.",
        "ingredients": [
            {"name": "suon non heo", "qty": 500, "unit": "g"},
            {"name": "thom nguyen trai", "qty": 200, "unit": "g"},
            {"name": "hanh tay", "qty": 50, "unit": "g"},
            {"name": "nuoc mam", "qty": 30, "unit": "ml"},
            {"name": "duong", "qty": 20, "unit": "g"},
            {"name": "dau mau dieu", "qty": 10, "unit": "ml"}
        ],
        "steps": ["Suon chat mieng, uop nuoc mam, duong.", "Thom cat mieng vua an.", "Kho suon voi thom, them dau mau dieu cho dep."],
        "cook_time": 30,
        "servings": 4,
        "tags": ["kho", "man", "dua com"],
        "diet": ["normal"],
        "device": ["noi"],
        "image": "suon-kho-thom.jpg",
        "region": "mien_nam",
        "nutrition": {"kcal": 500}
    },
    {
        "id": "ba-roi-chien-gion-20p",
        "title": "Ba roi heo chien gion 20 phut",
        "summary": "Ba roi chien gion rum, cham nuoc mam chanh ot.",
        "ingredients": [
            {"name": "ba roi heo", "qty": 500, "unit": "g"},
            {"name": "muoi", "qty": 5, "unit": "g"},
            {"name": "chanh khong hat", "qty": 1, "unit": "qua"},
            {"name": "ot hiem", "qty": 2, "unit": "trai"},
            {"name": "dau an", "qty": 50, "unit": "ml"}
        ],
        "steps": ["Ba roi luoc so, xam bi, xat muoi.", "Chien trong dau soi den khi vang gion.", "Pha nuoc cham chanh ot."],
        "cook_time": 20,
        "servings": 4,
        "tags": ["chien", "gion", "an nhau"],
        "diet": ["keto", "normal"],
        "device": ["chao"],
        "image": "ba-roi-chien.jpg",
        "region": "mien_bac",
        "nutrition": {"kcal": 600}
    },
    {
        "id": "thit-heo-xay-chung-trung-25p",
        "title": "Thit heo xay chung trung 25 phut",
        "summary": "Mon an dan da, mem mai, tre em rat thich.",
        "ingredients": [
            {"name": "thit heo xay", "qty": 200, "unit": "g"},
            {"name": "trung vit", "qty": 3, "unit": "qua"},
            {"name": "hanh ngo", "qty": 10, "unit": "g"},
            {"name": "tieu den", "qty": 3, "unit": "g"},
            {"name": "nuoc mam", "qty": 15, "unit": "ml"}
        ],
        "steps": ["Tron thit xay voi trung, nuoc mam, tieu, hanh.", "Chung cach thuy 20 phut den khi chin.", "Rac them tieu."],
        "cook_time": 25,
        "servings": 3,
        "tags": ["hap", "mem", "tre em"],
        "diet": ["normal"],
        "device": ["noi"],
        "image": "thit-chung-trung.jpg",
        "region": "mien_nam",
        "nutrition": {"kcal": 350}
    },
    {
        "id": "thit-dui-heo-luoc-20p",
        "title": "Thit dui heo luoc 20 phut",
        "summary": "Thit dui heo luoc mem, cuon banh trang voi rau song.",
        "ingredients": [
            {"name": "thit dui heo", "qty": 500, "unit": "g"},
            {"name": "xa lach bup mo", "qty": 200, "unit": "g"},
            {"name": "rau diep ca", "qty": 100, "unit": "g"},
            {"name": "nuoc mam", "qty": 30, "unit": "ml"}
        ],
        "steps": ["Thit dui rua sach, luoc chin mem.", "Thai lat mong.", "An kem rau song va nuoc mam."],
        "cook_time": 20,
        "servings": 4,
        "tags": ["luoc", "thanh dam", "cuon"],
        "diet": ["normal", "giam can"],
        "device": ["noi"],
        "image": "thit-dui-luoc.jpg",
        "region": "mien_bac",
        "nutrition": {"kcal": 400}
    },
    {
        "id": "canh-bi-xanh-thit-bam-15p",
        "title": "Canh bi xanh thit bam 15 phut",
        "summary": "Canh thanh mat, giai nhiet tu bi xanh va thit bam.",
        "ingredients": [
            {"name": "bi xanh", "qty": 400, "unit": "g"},
            {"name": "thit heo xay", "qty": 100, "unit": "g"},
            {"name": "hanh ngo", "qty": 10, "unit": "g"},
            {"name": "hat nem", "qty": 10, "unit": "g"}
        ],
        "steps": ["Xao so thit bam.", "Nau nuoc soi, cho bi xanh cat lat vao.", "Nem hat nem, rac hanh ngo."],
        "cook_time": 15,
        "servings": 4,
        "tags": ["canh", "thanh dam"],
        "diet": ["normal"],
        "device": ["noi"],
        "image": "canh-bi-xanh.jpg",
        "region": "mien_bac",
        "nutrition": {"kcal": 150}
    },

    # --- NHÓM GÀ (5 món) ---
    {
        "id": "ga-kho-nghe-30p",
        "title": "Ga kho nghe 30 phut",
        "summary": "Canh ga kho nghe vang uom, thom lung, tot cho da day.",
        "ingredients": [
            {"name": "canh ga", "qty": 500, "unit": "g"},
            {"name": "bot nghe", "qty": 10, "unit": "g"},
            {"name": "nuoc mam", "qty": 30, "unit": "ml"},
            {"name": "duong", "qty": 10, "unit": "g"}
        ],
        "steps": ["Ga chat mieng, uop bot nghe va gia vi.", "Kho liu riu cho tham va len mau dep.", "Nuoc set lai la duoc."],
        "cook_time": 30,
        "servings": 3,
        "tags": ["kho", "man", "dua com"],
        "diet": ["normal"],
        "device": ["noi"],
        "image": "ga-kho-nghe.jpg",
        "region": "mien_trung",
        "nutrition": {"kcal": 450}
    },
    {
        "id": "uc-ga-ap-chao-15p",
        "title": "Uc ga phi le ap chao 15 phut",
        "summary": "Mon an diet, healthy, uc ga mem khong kho.",
        "ingredients": [
            {"name": "uc ga phi le co da", "qty": 300, "unit": "g"},
            {"name": "tieu den", "qty": 5, "unit": "g"},
            {"name": "muoi", "qty": 3, "unit": "g"},
            {"name": "dau an", "qty": 10, "unit": "ml"},
            {"name": "xa lach bup mo", "qty": 100, "unit": "g"}
        ],
        "steps": ["Uc ga uop muoi tieu.", "Ap chao chin vang 2 mat.", "An kem xa lach."],
        "cook_time": 15,
        "servings": 2,
        "tags": ["chien", "healthy", "giam can"],
        "diet": ["eat clean", "diet"],
        "device": ["chao"],
        "image": "uc-ga-ap-chao.jpg",
        "region": "quoc_te",
        "nutrition": {"kcal": 250}
    },
    {
        "id": "long-ga-xao-muop-15p",
        "title": "Long ga xao muop huong 15 phut",
        "summary": "Long ga gion, muop huong thom ngot.",
        "ingredients": [
            {"name": "long ga", "qty": 300, "unit": "g"},
            {"name": "muop huong", "qty": 2, "unit": "trai"},
            {"name": "hanh ngo", "qty": 10, "unit": "g"},
            {"name": "hat nem", "qty": 5, "unit": "g"}
        ],
        "steps": ["Long ga lam sach, xao chin.", "Cho muop vao xao nhanh tay.", "Nem gia vi vua an, rac hanh."],
        "cook_time": 15,
        "servings": 3,
        "tags": ["xao", "nhanh", "dan da"],
        "diet": ["normal"],
        "device": ["chao"],
        "image": "long-ga-xao-muop.jpg",
        "region": "mien_bac",
        "nutrition": {"kcal": 200}
    },
    {
        "id": "tim-ga-xao-ot-chuong-20p",
        "title": "Tim ga xao ot chuong 20 phut",
        "summary": "Tim ga xao ot chuong gion ngot, mau sac bat mat.",
        "ingredients": [
            {"name": "tim ga", "qty": 300, "unit": "g"},
            {"name": "ot chuong", "qty": 2, "unit": "trai"},
            {"name": "hanh tay", "qty": 1, "unit": "cu"},
            {"name": "sa te", "qty": 10, "unit": "g"}
        ],
        "steps": ["Tim ga xao san voi sa te.", "Cho ot chuong, hanh tay vao dao deu.", "Nem nem vua an."],
        "cook_time": 20,
        "servings": 3,
        "tags": ["xao", "cay", "nhanh"],
        "diet": ["normal"],
        "device": ["chao"],
        "image": "tim-ga-xao.jpg",
        "region": "mien_nam",
        "nutrition": {"kcal": 250}
    },
    {
        "id": "chan-ga-hap-hanh-20p",
        "title": "Chan ga hap hanh 20 phut",
        "summary": "Mon nhau lai rai, chan ga gion, thom mui hanh.",
        "ingredients": [
            {"name": "chan ga", "qty": 500, "unit": "g"},
            {"name": "hanh tay", "qty": 100, "unit": "g"},
            {"name": "hanh ngo", "qty": 50, "unit": "g"},
            {"name": "muoi", "qty": 5, "unit": "g"},
            {"name": "tieu den", "qty": 3, "unit": "g"}
        ],
        "steps": ["Chan ga lam sach, luoc so.", "Hap chan ga voi hanh tay, hanh la.", "Cham muoi tieu chanh."],
        "cook_time": 20,
        "servings": 4,
        "tags": ["hap", "an choi", "nhau"],
        "diet": ["normal"],
        "device": ["noi"],
        "image": "chan-ga-hap.jpg",
        "region": "mien_bac",
        "nutrition": {"kcal": 300}
    },

    # --- NHÓM BÒ (4 món) ---
    {
        "id": "bo-luc-lac-25p",
        "title": "Bo luc lac 25 phut",
        "summary": "Dui bo cat khoi xao ot chuong, hanh tay, dam da.",
        "ingredients": [
            {"name": "dui bo", "qty": 400, "unit": "g"},
            {"name": "ot chuong", "qty": 150, "unit": "g"},
            {"name": "hanh tay", "qty": 100, "unit": "g"},
            {"name": "dau hao", "qty": 20, "unit": "ml"},
            {"name": "ca chua", "qty": 1, "unit": "qua"}
        ],
        "steps": ["Bo uop gia vi, xao lua lon.", "Xao rau cu chin toi.", "Tron bo va rau cu, tat bep."],
        "cook_time": 25,
        "servings": 3,
        "tags": ["xao", "tiec", "ngon mieng"],
        "diet": ["normal"],
        "device": ["chao"],
        "image": "bo-luc-lac.jpg",
        "region": "quoc_te",
        "nutrition": {"kcal": 450}
    },
    {
        "id": "ba-chi-bo-cuon-dau-bap-20p",
        "title": "Ba chi bo cuon dau bap nuong 20 phut",
        "summary": "Ba chi bo beo ngay cuon dau bap nuong sa te.",
        "ingredients": [
            {"name": "ba chi bo", "qty": 300, "unit": "g"},
            {"name": "dau bap", "qty": 200, "unit": "g"},
            {"name": "sa te", "qty": 20, "unit": "g"},
            {"name": "dau an", "qty": 10, "unit": "ml"}
        ],
        "steps": ["Cuon thit bo quanh dau bap.", "Quet sa te.", "Nuong hoac ap chao chin vang."],
        "cook_time": 20,
        "servings": 3,
        "tags": ["nuong", "an choi", "tiec"],
        "diet": ["keto", "normal"],
        "device": ["chao"],
        "image": "bo-cuon-dau-bap.jpg",
        "region": "mien_nam",
        "nutrition": {"kcal": 500}
    },
    {
        "id": "canh-bo-nau-ngot-20p",
        "title": "Canh bo nau ngot (nau ca chua) 20 phut",
        "summary": "Canh bo nau voi ca chua va thom, chua thanh de an.",
        "ingredients": [
            {"name": "ba chi bo", "qty": 200, "unit": "g"},
            {"name": "ca chua", "qty": 2, "unit": "qua"},
            {"name": "thom nguyen trai", "qty": 100, "unit": "g"},
            {"name": "hanh ngo", "qty": 10, "unit": "g"}
        ],
        "steps": ["Xao ca chua va thom.", "Do nuoc soi, tha thit bo vao.", "Nem gia vi, rac hanh ngo, an nong."],
        "cook_time": 20,
        "servings": 3,
        "tags": ["canh", "nhanh", "chua ngot"],
        "diet": ["normal"],
        "device": ["noi"],
        "image": "canh-bo-nau-ngot.jpg",
        "region": "mien_nam",
        "nutrition": {"kcal": 300}
    },
    {
        "id": "bo-xao-ca-tim-20p",
        "title": "Bo xao ca tim 20 phut",
        "summary": "Dui bo xao ca tim mem ngot, la mieng.",
        "ingredients": [
            {"name": "dui bo", "qty": 200, "unit": "g"},
            {"name": "ca tim", "qty": 2, "unit": "trai"},
            {"name": "hanh ngo", "qty": 10, "unit": "g"},
            {"name": "dau hao", "qty": 15, "unit": "ml"}
        ],
        "steps": ["Ca tim cat khuc, ngam nuoc muoi, xao mem.", "Xao thit bo chin toi.", "Dao chung bo va ca tim, nem dau hao."],
        "cook_time": 20,
        "servings": 3,
        "tags": ["xao", "com nha"],
        "diet": ["normal"],
        "device": ["chao"],
        "image": "bo-xao-ca-tim.jpg",
        "region": "mien_bac",
        "nutrition": {"kcal": 350}
    },

    # --- NHÓM HẢI SẢN (6 món) ---
    {
        "id": "ca-dieu-hong-sot-ca-25p",
        "title": "Ca dieu hong sot ca chua 25 phut",
        "summary": "Ca dieu hong chien gion, sot ca chua dam da.",
        "ingredients": [
            {"name": "ca dieu hong lam sach", "qty": 800, "unit": "g"},
            {"name": "ca chua", "qty": 3, "unit": "qua"},
            {"name": "nuoc mam", "qty": 20, "unit": "ml"},
            {"name": "hanh ngo", "qty": 10, "unit": "g"}
        ],
        "steps": ["Ca chien vang 2 mat.", "Lam sot ca chua min.", "Ruoi sot len ca hoac rim so cho tham."],
        "cook_time": 25,
        "servings": 4,
        "tags": ["man", "dua com", "chien"],
        "diet": ["normal"],
        "device": ["chao"],
        "image": "ca-sot-ca.jpg",
        "region": "mien_nam",
        "nutrition": {"kcal": 400}
    },
    {
        "id": "canh-chua-ca-basa-30p",
        "title": "Canh chua ca basa 30 phut",
        "summary": "Canh chua ca basa beo ngay voi thom, ca chua, dau bap.",
        "ingredients": [
            {"name": "ca basa cat khuc", "qty": 500, "unit": "g"},
            {"name": "thom nguyen trai", "qty": 100, "unit": "g"},
            {"name": "dau bap", "qty": 100, "unit": "g"},
            {"name": "ca chua", "qty": 2, "unit": "qua"},
            {"name": "chanh khong hat", "qty": 1, "unit": "qua"}
        ],
        "steps": ["Nau nuoc soi, cho ca vao.", "Them thom, ca chua, dau bap.", "Nem chua ngot voi chanh, duong, nuoc mam."],
        "cook_time": 30,
        "servings": 4,
        "tags": ["canh", "chua ngot", "thanh mat"],
        "diet": ["normal"],
        "device": ["noi"],
        "image": "canh-chua-basa.jpg",
        "region": "mien_nam",
        "nutrition": {"kcal": 350}
    },
    {
        "id": "ca-nuc-kho-tieu-40p",
        "title": "Ca nuc kho tieu 40 phut",
        "summary": "Ca nuc kho cung, tham gia vi, an voi com trang.",
        "ingredients": [
            {"name": "ca nuc lam sach", "qty": 500, "unit": "g"},
            {"name": "tieu den", "qty": 5, "unit": "g"},
            {"name": "nuoc mam", "qty": 40, "unit": "ml"},
            {"name": "duong", "qty": 20, "unit": "g"},
            {"name": "dau mau dieu", "qty": 10, "unit": "ml"}
        ],
        "steps": ["Uop ca voi gia vi 15 phut.", "Kho lua nho den khi nuoc keo lai.", "Rac them tieu."],
        "cook_time": 40,
        "servings": 4,
        "tags": ["kho", "man", "truyen thong"],
        "diet": ["normal"],
        "device": ["noi"],
        "image": "ca-nuc-kho.jpg",
        "region": "mien_trung",
        "nutrition": {"kcal": 400}
    },
    {
        "id": "tom-rim-ba-roi-25p",
        "title": "Tom rim thit ba roi 25 phut",
        "summary": "Tom rang thit ba chi man ngot, dam da.",
        "ingredients": [
            {"name": "tom the", "qty": 300, "unit": "g"},
            {"name": "ba roi heo", "qty": 200, "unit": "g"},
            {"name": "duong", "qty": 20, "unit": "g"},
            {"name": "nuoc mam", "qty": 20, "unit": "ml"}
        ],
        "steps": ["Thit ba chi dao chay canh.", "Cho tom vao dao cung.", "Nem nuoc mam, duong, rim can kho."],
        "cook_time": 25,
        "servings": 4,
        "tags": ["rim", "man", "dua com"],
        "diet": ["normal"],
        "device": ["chao"],
        "image": "tom-rim-thit.jpg",
        "region": "mien_bac",
        "nutrition": {"kcal": 500}
    },
    {
        "id": "muc-xao-sa-te-15p",
        "title": "Muc xao sa te thap cam 15 phut",
        "summary": "Muc xao cay nong voi ot chuong, dua leo, sa te.",
        "ingredients": [
            {"name": "muc nut lam sach", "qty": 400, "unit": "g"},
            {"name": "sa te", "qty": 20, "unit": "g"},
            {"name": "ot chuong", "qty": 1, "unit": "trai"},
            {"name": "dua leo", "qty": 1, "unit": "trai"},
            {"name": "hanh tay", "qty": 1, "unit": "cu"}
        ],
        "steps": ["Muc xao san voi sa te.", "Cho rau cu vao xao chin toi.", "Nem gia vi vua an."],
        "cook_time": 15,
        "servings": 3,
        "tags": ["xao", "cay", "nhanh"],
        "diet": ["normal"],
        "device": ["chao"],
        "image": "muc-xao-sa-te.jpg",
        "region": "mien_nam",
        "nutrition": {"kcal": 300}
    },
    {
        "id": "mi-xao-hai-san-15p",
        "title": "Mi xao hai san 15 phut",
        "summary": "Mi xao voi tom, muc va rau cai, nhanh gon cho bua trua.",
        "ingredients": [
            {"name": "mi an lien kangshifu", "qty": 2, "unit": "goi"},
            {"name": "tom the", "qty": 100, "unit": "g"},
            {"name": "rau muc", "qty": 100, "unit": "g"},
            {"name": "cai ngot", "qty": 200, "unit": "g"}
        ],
        "steps": ["Trung mi so qua.", "Xao tom, muc va rau.", "Cho mi vao dao nhanh, nem gia vi."],
        "cook_time": 15,
        "servings": 2,
        "tags": ["xao", "an nhanh", "tien loi"],
        "diet": ["normal"],
        "device": ["chao"],
        "image": "mi-xao-hai-san.jpg",
        "region": "quoc_te",
        "nutrition": {"kcal": 550}
    },

    # --- NHÓM RAU & CANH KHÁC (5 món) ---
    {
        "id": "canh-khoai-tay-ca-rot-suon-30p",
        "title": "Canh khoai tay ca rot suon non 30 phut",
        "summary": "Canh rau cu ham suon ngot nuoc, bo duong.",
        "ingredients": [
            {"name": "suon non heo", "qty": 300, "unit": "g"},
            {"name": "khoai tay", "qty": 300, "unit": "g"},
            {"name": "ca rot", "qty": 200, "unit": "g"},
            {"name": "hanh ngo", "qty": 10, "unit": "g"}
        ],
        "steps": ["Suon ham mem.", "Cho ca rot, khoai tay vao nau chin.", "Nem gia vi, rac hanh."],
        "cook_time": 30,
        "servings": 4,
        "tags": ["canh", "bo duong"],
        "diet": ["normal"],
        "device": ["noi"],
        "image": "canh-suon-khoai-tay.jpg",
        "region": "mien_bac",
        "nutrition": {"kcal": 400}
    },
    {
        "id": "canh-rau-ngot-tom-15p",
        "title": "Canh rau ngot nau tom 15 phut",
        "summary": "Canh rau ngot nau tom tuoi ngot lanh.",
        "ingredients": [
            {"name": "rau ngot", "qty": 300, "unit": "g"},
            {"name": "tom the", "qty": 100, "unit": "g"},
            {"name": "hat nem", "qty": 10, "unit": "g"}
        ],
        "steps": ["Tom bam nho, xao so.", "Do nuoc soi, cho rau ngot vao.", "Nem hat nem vua an."],
        "cook_time": 15,
        "servings": 4,
        "tags": ["canh", "thanh dam", "rau xanh"],
        "diet": ["normal"],
        "device": ["noi"],
        "image": "canh-rau-ngot.jpg",
        "region": "mien_bac",
        "nutrition": {"kcal": 200}
    },
    {
        "id": "cai-ngot-xao-long-ga-15p",
        "title": "Cai ngot xao long ga 15 phut",
        "summary": "Mon xao nhanh, du chat xo va dam.",
        "ingredients": [
            {"name": "cai ngot", "qty": 400, "unit": "g"},
            {"name": "long ga", "qty": 200, "unit": "g"},
            {"name": "tieu den", "qty": 3, "unit": "g"},
            {"name": "dau an", "qty": 15, "unit": "ml"}
        ],
        "steps": ["Long ga xao chin toi.", "Cho cai ngot vao xao nhanh.", "Nem gia vi, rac tieu."],
        "cook_time": 15,
        "servings": 3,
        "tags": ["xao", "nhanh", "rau xanh"],
        "diet": ["normal"],
        "device": ["chao"],
        "image": "cai-ngot-xao.jpg",
        "region": "mien_bac",
        "nutrition": {"kcal": 250}
    },
    {
        "id": "rau-muong-xao-10p",
        "title": "Rau muong xao toi 10 phut",
        "summary": "Rau muong xao xanh muot, gion gion.",
        "ingredients": [
            {"name": "rau muong", "qty": 500, "unit": "g"},
            {"name": "dau an", "qty": 20, "unit": "ml"},
            {"name": "muoi", "qty": 5, "unit": "g"},
            {"name": "nuoc mam", "qty": 5, "unit": "ml"}
        ],
        "steps": ["Rau luoc so qua nuoc soi.", "Xao nhanh voi dau an lua lon.", "Nem muoi hoac nuoc mam."],
        "cook_time": 10,
        "servings": 3,
        "tags": ["xao", "nhanh", "chay"],
        "diet": ["chay", "normal"],
        "device": ["chao"],
        "image": "rau-muong-xao.jpg",
        "region": "mien_bac",
        "nutrition": {"kcal": 150}
    },
    {
        "id": "canh-trung-ca-chua-10p",
        "title": "Canh trung ca chua 10 phut",
        "summary": "Canh may (canh trung) nau nhanh, de an.",
        "ingredients": [
            {"name": "trung ga", "qty": 2, "unit": "qua"},
            {"name": "ca chua", "qty": 2, "unit": "qua"},
            {"name": "hanh ngo", "qty": 10, "unit": "g"},
            {"name": "dau an", "qty": 10, "unit": "ml"}
        ],
        "steps": ["Xao ca chua mem.", "Do nuoc soi.", "Do trung vao khuay deu tao van.", "Rac hanh ngo."],
        "cook_time": 10,
        "servings": 3,
        "tags": ["canh", "nhanh", "tiet kiem"],
        "diet": ["normal"],
        "device": ["noi"],
        "image": "canh-trung.jpg",
        "region": "mien_bac",
        "nutrition": {"kcal": 150}
    },

    # --- MÓN NHANH / TIỆN LỢI (5 món) ---
    {
        "id": "bun-gio-heo-15p",
        "title": "Bun gio heo nau nhanh 15 phut",
        "summary": "Bun gio heo an lien them thit dui heo luoc.",
        "ingredients": [
            {"name": "bun gio heo hang nga", "qty": 2, "unit": "goi"},
            {"name": "thit dui heo", "qty": 100, "unit": "g"},
            {"name": "gia", "qty": 50, "unit": "g"},
            {"name": "chanh khong hat", "qty": 1, "unit": "qua"}
        ],
        "steps": ["Thit dui luoc chin, thai mong.", "Nau bun theo huong dan goi.", "Them thit va rau an kem."],
        "cook_time": 15,
        "servings": 2,
        "tags": ["an sang", "nhanh", "tien loi"],
        "diet": ["normal"],
        "device": ["noi"],
        "image": "bun-gio-heo.jpg",
        "region": "mien_nam",
        "nutrition": {"kcal": 400}
    },
    {
        "id": "trung-chien-hanh-tay-10p",
        "title": "Trung chien hanh tay 10 phut",
        "summary": "Trung chien hanh tay thom ngot, don gian.",
        "ingredients": [
            {"name": "trung vit", "qty": 3, "unit": "qua"},
            {"name": "hanh tay", "qty": 100, "unit": "g"},
            {"name": "nuoc mam", "qty": 10, "unit": "ml"},
            {"name": "tieu den", "qty": 2, "unit": "g"}
        ],
        "steps": ["Hanh tay thai mong, xao so.", "Danh trung voi gia vi.", "Do trung vao chien chin vang."],
        "cook_time": 10,
        "servings": 3,
        "tags": ["chien", "nhanh", "tiet kiem"],
        "diet": ["normal"],
        "device": ["chao"],
        "image": "trung-chien-hanh.jpg",
        "region": "mien_bac",
        "nutrition": {"kcal": 300}
    },
    {
        "id": "rau-den-luoc-10p",
        "title": "Rau den luoc 10 phut",
        "summary": "Rau den luoc thanh mat, nuoc canh mau do dep.",
        "ingredients": [
            {"name": "rau den", "qty": 500, "unit": "g"},
            {"name": "muoi", "qty": 3, "unit": "g"}
        ],
        "steps": ["Rau nhat sach.", "Luoc nuoc soi them chut muoi.", "Vot ra dia, lay nuoc lam canh."],
        "cook_time": 10,
        "servings": 3,
        "tags": ["luoc", "chay", "thanh dam"],
        "diet": ["chay", "diet"],
        "device": ["noi"],
        "image": "rau-den-luoc.jpg",
        "region": "mien_bac",
        "nutrition": {"kcal": 50}
    },
    {
        "id": "salad-tron-dau-giam-10p",
        "title": "Salad tron dau giam 10 phut",
        "summary": "Salad rau cu tuoi mat, sot dau giam chua ngot.",
        "ingredients": [
            {"name": "xa lach bup mo", "qty": 200, "unit": "g"},
            {"name": "ca chua", "qty": 2, "unit": "qua"},
            {"name": "dua leo", "qty": 1, "unit": "trai"},
            {"name": "dau an", "qty": 20, "unit": "ml"},
            {"name": "chanh khong hat", "qty": 1, "unit": "qua"},
            {"name": "duong", "qty": 15, "unit": "g"}
        ],
        "steps": ["Rau cu rua sach, cat mieng.", "Pha sot: dau an, nuoc cot chanh, duong, muoi.", "Tron deu truoc khi an."],
        "cook_time": 10,
        "servings": 2,
        "tags": ["tron", "healthy", "giam can"],
        "diet": ["chay", "diet", "eat clean"],
        "device": ["bat"],
        "image": "salad-dau-giam.jpg",
        "region": "quoc_te",
        "nutrition": {"kcal": 150}
    },
    {
        "id": "muc-hap-hanh-15p",
        "title": "Muc nut hap hanh tay 15 phut",
        "summary": "Muc nut ngot gion hap hanh tay thom lung.",
        "ingredients": [
            {"name": "muc nut lam sach", "qty": 500, "unit": "g"},
            {"name": "hanh tay", "qty": 150, "unit": "g"},
            {"name": "hanh ngo", "qty": 20, "unit": "g"},
            {"name": "nuoc mam", "qty": 20, "unit": "ml"}
        ],
        "steps": ["Muc rua sach.", "Xep hanh tay, hanh ngo len tren.", "Hap 10 phut, cham nuoc mam."],
        "cook_time": 15,
        "servings": 4,
        "tags": ["hap", "thanh dam", "nhanh"],
        "diet": ["diet", "keto"],
        "device": ["noi"],
        "image": "muc-hap-hanh.jpg",
        "region": "mien_trung",
        "nutrition": {"kcal": 200}
    }
]

# Ghi file
print(f"Dang tao {len(recipes)} file recipes tu nguyen lieu trong kho...")
count = 0
for r in recipes:
    filename = f"{r['id']}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(r, f, ensure_ascii=False, indent=2)
    count += 1

print(f"✅ Xong! Da tao {count} file tai {OUTPUT_DIR}")
print("👉 QUAN TRONG: Hay chay lenh: python serverAI/features/build_index.py de cap nhat he thong.")