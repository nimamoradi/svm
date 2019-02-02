from svm_module.srt_to_txt import main


def srt(path):
    import glob

    list_files = glob.glob(path + "/*.srt")
    print(len(list_files))

    for target_list in list_files:
        main(target_list)

    list_files = glob.glob(path + "/*.txt")
    print(len(list_files))
