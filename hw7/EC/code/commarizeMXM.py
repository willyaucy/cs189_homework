import sys

filename = sys.argv[1]
with open(filename, "r") as f:
    lastpos = None
    while True:
        line = f.readline()
        #skip lines preceding with a hash or percent symbol
        if line[0] == "#" or line[0] == "%":
            lastpos = f.tell()
        else:
            break
    if lastpos != None:
        f.seek(lastpos-1)
        
    for line in f.readlines():
        output = "" #empty string
        fields = line.strip().split(',')[2:]
        curr = 0
        try:
            for i in range(5000):
                if int(fields[curr].split(':')[0]) != (i+1):
                    output += "," #add a delimiter
                else:
                    freq = fields[curr].split(':')[1]
                    output += freq + ","
                    curr += 1
        except IndexError:
            pass #no more sparse indices
        extraCommas = ((5000-int(fields[curr-1].split(':')[0])))
        output += "".join([","]) * extraCommas
        output = output[:-1] + "\n"
        with open(filename.split('.')[0] + ".csv", "a") as f2:
            f2.write(output)
