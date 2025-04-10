import h5py

ipr_path = r"C:\Users\zoujialin\Desktop\gold\demo\C03834C6\C03834C6_SC_20240808_130306_4.0.0.ipr"

with h5py.File(ipr_path) as conf:
    if "HE" in conf:
        ssdna_group = conf["HE"]
        if "Register" in ssdna_group:
            register_group = ssdna_group["Register"]

            # track_points = conf["HE"]["Stitch"]["TemplatePoint"][...]
            # print(track_points)

            if "CounterRot90" in register_group.attrs:    # 不同Name对应的Value在这里修改
                counter_rot90_value = register_group.attrs["CounterRot90"]
                print(counter_rot90_value)
            else:
                print("CounterRot90 not found in Register's attributes.")
        else:
            print("Register group not found in ssDNA.")
    else:
        print("ssDNA group not found in the file.")