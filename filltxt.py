f = open("prod_id.txt", "w")
for i in range(7001, 7008):
    f.write(str(i).zfill(6) + "_1.png")
    f.write("\n")
f.close()