
import datetime

class ReadData(object):
    def __init__(self):

        self.data = {}

        self.data2 = {}

        self.data3 = {}

        self.data4 = {}

    def readAddress(self, fileName):
        f = open(fileName, 'r')
        attri = f.readline().strip().split(',')
        attr = []
        for i in attri:
            attr.append(i)
        self.address = {}   
        self.zip = {}       
        self.changeAddress = set()   
        self.move_moveBack = set()  
        for i in f:
            item = i.strip().split(',')
            ads = (item[1], item[2], item[3], item[4], item[5])
            if ads in self.address:
                if item[0] not in self.address[ads]:
                    self.address[ads].append(item[0])
            else:
                studyID = []
                studyID.append(item[0])
                self.address[ads] = studyID
            if item[0] in self.zip:
                if item[5] not in self.zip[item[0]]:
                    self.zip[item[0]].append(item[5])
            else:
                zipCode = []
                zipCode.append(item[5])
                self.zip[item[0]] = zipCode
            if item[0] in self.data:
                adds = (item[1], item[2], item[3], item[4], item[5])
                year = [item[6], item[7]]
                if adds in self.data[item[0]]:
                    overlap = False
                    for year_range in self.data[item[0]][adds]:
                        y1 = year_range[0]
                        y2 = year_range[1]
                        y3 = item[6]
                        y4 = item[7]
                        if y1 =='':
                            y1 = 0
                        else:
                            y1 = int(y1)
                        if y2 =='':
                            y2 = datetime.datetime.now().year
                        else:
                            y2 = int(y2)
                        if y3 == '':
                            y3 = 0
                        else:
                            y3 = int(y3)
                        if y4 == '':
                            y4 = datetime.datetime.now().year
                        else:
                            y4 = int(y4)
                        #   check if two periods overlap
                        if y2 < y3-1:
                            continue
                        if y4 < y1-1:
                            continue
                        overlap = True
                        mi = min([y1, y2, y3, y4])
                        ma = max([y1, y2, y3, y4])
                        if mi == 0 and ma == datetime.datetime.now().year:
                            #year = ('', '')
                            #self.data[item[0]][0][adds] = year
                            year_range[0] = ''
                            year_range[1] = ''
                        elif mi != 0 and ma == datetime.datetime.now().year:
                            #year = (mi, '')
                            #self.data[item[0]][0][adds] = year
                            year_range[0] = mi
                            year_range[1] = ''
                        elif ma != datetime.datetime.now().year and mi == 0:
                            #year = ('', ma)
                            #self.data[item[0]][0][adds] = year
                            year_range[0] = ''
                            year_range[1] = ma
                        else:
                            #year = (mi, ma)
                            #self.data[item[0]][0][adds] = year
                            year_range[0] = mi
                            year_range[1] = ma
                    if not overlap:
                        self.data[item[0]][adds].append(year)
                        self.move_moveBack.add(item[0])
                else:
                    self.data[item[0]][adds] = [year]
                    self.changeAddress.add(item[0])
            else:
                #lis = []
                addr = {}
                adds = (item[1], item[2], item[3], item[4], item[5])
                year = [item[6], item[7]]
                addr[adds] = [year]
                #lis.append(addr)
                #self.data[item[0]] = lis
                self.data[item[0]] = addr
        f.flush()
        f.close()

    def readName(self, fileName):
        f = open(fileName, 'r')
        attri = f.readline().strip().split(',')
        attr = []
        for i in attri:
            attr.append(i)
        self.lastName = {}  #   map study ID to last name with different time period
        self.changeName = set() #   a set of study ID who changed last name
        self.nameChangeBack = set() #   a set of study who changed last name and changed back
        for i in f:
            item = i.strip().split(',')
            if item[0] in self.lastName:
                year = [item[4], item[5]]
                if item[1] in self.lastName[item[0]]:
                    overlap = False
                    for year_range in self.lastName[item[0]][item[1]]:
                        y1 = year_range[0]
                        y2 = year_range[1]
                        y3 = item[4]
                        y4 = item[5]
                        if y1 == '':
                            y1 = 0
                        else:
                            y1 = int(y1)
                        if y2 == '':
                            y2 = datetime.datetime.now().year
                        else:
                            y2 = int(y2)
                        if y3 == '':
                            y3 = 0
                        else:
                            y3 = int(y3)
                        if y4 == '':
                            y4 = datetime.datetime.now().year
                        else:
                            y4 = int(y4)
                        #   check if two periods overlap
                        if y2 < y3-1:
                            continue
                        if y4 < y1-1:
                            continue
                        overlap = True
                        mi = min([y1, y2, y3, y4])
                        ma = max([y1, y2, y3, y4])
                        if mi == 0 and ma == datetime.datetime.now().year:
                            #year = ('', '')
                            #self.lastName[item[0]][item[1]] = year
                            year_range[0] = ''
                            year_range[1] = ''
                        elif mi != 0 and ma == datetime.datetime.now().year:
                            #year = (mi, '')
                            #self.lastName[item[0]][item[1]] = year
                            year_range[0] = mi
                            year_range[1] = ''
                        elif ma != datetime.datetime.now().year and mi == 0:
                            #year = ('', ma)
                            #self.lastName[item[0]][item[1]] = year
                            year_range = ''
                            year_range = ma
                        else:
                            #year = (mi, ma)
                            #self.lastName[item[0]][item[1]] = year
                            year_range[0] = mi
                            year_range[1] = ma
                    if not overlap:
                        self.lastName[item[0]][item[1]].append(year)
                        self.nameChangeBack.add(item[0])
                else:
                    self.lastName[item[0]][item[1]] = [year]
                    self.changeName.add(item[0])
            else:
                na = {}
                year = [item[4], item[5]]
                na[item[1]] = [year]
                self.lastName[item[0]] = na

            if item[0] in self.data2:
                names = (item[1], item[2], item[3])
                year = [item[4], item[5]]
                if names in self.data2[item[0]]:
                    overlap = False
                    for year_range in self.data2[item[0]][names]:
                        y1 = year_range[0]
                        y2 = year_range[1]
                        y3 = item[4]
                        y4 = item[5]
                        if y1 =='':
                            y1 = 0
                        else:
                            y1 = int(y1)
                        if y2 =='':
                            y2 = datetime.datetime.now().year
                        else:
                            y2 = int(y2)
                        if y3 == '':
                            y3 = 0
                        else:
                            y3 = int(y3)
                        if y4 == '':
                            y4 = datetime.datetime.now().year
                        else:
                            y4 = int(y4)
                        #   check if two periods overlap
                        if y2 < y3-1:
                            continue
                        if y4 < y1-1:
                            continue
                        overlap = True
                        mi = min([y1, y2, y3, y4])
                        ma = max([y1, y2, y3, y4])
                        if mi == 0 and ma == datetime.datetime.now().year:
                            #year = ('', '')
                            #self.data2[item[0]][0][names] = year
                            year_range[0] = ''
                            year_range[1] = ''
                        elif mi != 0 and ma == datetime.datetime.now().year:
                            #year = (mi, '')
                            #self.data2[item[0]][0][names] = year
                            year_range[0] = mi
                            year_range[1] = ''
                        elif ma != datetime.datetime.now().year and mi == 0:
                            #year = ('', ma)
                            #self.data2[item[0]][0][names] = year
                            year_range[0] = ''
                            year_range[1] = ma
                        else:
                            #year = (mi, ma)
                            #self.data2[item[0]][0][names] = year
                            year_range[0] = mi
                            year_range[1] = ma
                    if not overlap:
                        self.data2[item[0]][names].append(year)
                        #self.nameChangeBack.add(item[0])
                else:
                    self.data2[item[0]][names] = [year]
                    #self.changeName.add(item[0])
            else:
                #lis = []
                name = {}
                names = (item[1], item[2], item[3])
                year = [item[4], item[5]]
                name[names] = [year]
                #lis.append(name)
                self.data2[item[0]] = name
        f.flush()
        f.close()

    def readDemo(self, fileName):
        f = open(fileName, 'r')
        attri = f.readline().strip().split(',')
        attr = []
        for i in attri:
            attr.append(i)
        self.age = {}       #   map study ID to age
        self.dob = {}       #   map study ID to date of birth
        self.dateOfDeceased = {}    #   map study ID to date of deceased
        self.gender = {}    #   map study ID to gender
        self.phone = {}     #   map study ID to phone number
        self.changeNumber = set()  #   a set of study ID changed contact number on EHR record
        for i in f:
            item = i.strip().split(',')
            if item[0] not in self.age:
                if item[2] != '':
                    if item[3] != '':
                        self.age[item[0]] = int(item[3]) - int(item[2])
                    else:
                        self.age[item[0]] = datetime.datetime.now().year - int(item[2])
                else:
                    self.age[item[0]] = item[2]
            if item[0] not in self.gender:
                if item[1] and (item[1].upper() == 'M' or item[1].upper() == 'F'):
                    self.gender[item[0]] = item[1]
            if item[0] not in self.dob:
                if item[2] != '':
                    self.dob[item[0]] = item[2]
            if item[0] not in self.dateOfDeceased:
                if item[3] != '':
                    self.dateOfDeceased[item[0]] = item[3]
            if item[0] in self.phone:
                year = [item[5], item[6]]
                if item[4] in self.phone[item[0]]:
                    overlap = False
                    for year_range in self.phone[item[0]][item[4]]:
                        y1 = year_range[0]
                        y2 = year_range[1]
                        y3 = item[5]
                        y4 = item[6]
                        if y1 == '':
                            y1 = 0
                        else:
                            y1 = int(y1)
                        if y2 == '':
                            y2 = datetime.datetime.now().year
                        else:
                            y2 = int(y2)
                        if y3 == '':
                            y3 = 0
                        else:
                            y3 = int(y3)
                        if y4 == '':
                            y4 = datetime.datetime.now().year
                        else:
                            y4 = int(y4)
                        if y2 < y3-1:
                            continue
                        if y4 < y1-1:
                            continue
                        overlap = True
                        mi = min([y1, y2, y3, y4])
                        ma = max([y1, y2, y3, y4])
                        if mi == 0 and ma == datetime.datetime.now().year:
                            #year = ('', '')
                            #self.phone[item[0]][item[4]] = year
                            year_range[0] = ''
                            year_range[1] = ''
                        elif mi != 0 and ma == datetime.datetime.now().year:
                            #year = (mi, '')
                            #self.phone[item[0]][item[4]] = year
                            year_range[0] = mi
                            year_range[1] = ''
                        elif ma != datetime.datetime.now().year and mi == 0:
                            #year = ('', ma)
                            #self.phone[item[0]][item[4]] = year
                            year_range = ''
                            year_range = ma
                        else:
                            #year = (mi, ma)
                            #self.phone[item[0]][item[4]] = year
                            year_range[0] = mi
                            year_range[1] = ma
                    if not overlap:
                        self.phone[item[0]][item[4]].append(year)
                else:
                    self.phone[item[0]][item[4]] = [year]
                    self.changeNumber.add(item[0])
            else:
                pho = {}
                year = [item[5], item[6]]
                pho[item[4]] = [year]
                self.phone[item[0]] = pho

            if item[0] in self.data3:
                info = (item[1], item[2], item[3], item[4])
                year = [item[5], item[6]]
                if info in self.data3[item[0]]:
                    overlap = False
                    for year_range in self.data3[item[0]][info]:
                        y1 = year_range[0]
                        y2 = year_range[1]
                        y3 = item[5]
                        y4 = item[6]
                        if y1 =='':
                            y1 = 0
                        else:
                            y1 = int(y1)
                        if y2 =='':
                            y2 = datetime.datetime.now().year
                        else:
                            y2 = int(y2)
                        if y3 == '':
                            y3 = 0
                        else:
                            y3 = int(y3)
                        if y4 == '':
                            y4 = datetime.datetime.now().year
                        else:
                            y4 = int(y4)
                        #   check if two periods overlap
                        if y2 < y3-1:
                            continue
                        if y4 < y1-1:
                            continue
                        overlap = True
                        mi = min([y1, y2, y3, y4])
                        ma = max([y1, y2, y3, y4])
                        if mi == 0 and ma == datetime.datetime.now().year:
                            #year = ('', '')
                            #self.data3[item[0]][0][info] = year
                            year_range[0] = ''
                            year_range[1] = ''
                        elif mi != 0 and ma == datetime.datetime.now().year:
                            #year = (mi, '')
                            #self.data3[item[0]][0][info] = year
                            year_range[0] = mi
                            year_range[1] = ''
                        elif ma != datetime.datetime.now().year and mi == 0:
                            #year = ('', ma)
                            #self.data3[item[0]][0][info] = year
                            year_range[0] = ''
                            year_range[1] = ma
                        else:
                            #year = (mi, ma)
                            #self.data3[item[0]][0][info] = year
                            year_range[0] = mi
                            year_range[1] = ma
                    if not overlap:
                        self.data3[item[0]][info].append(year)
                else:
                    self.data3[item[0]][info] = [year]
            else:
                #lis = []
                demo = {}
                info = (item[1], item[2], item[3], item[4])
                year = [item[5], item[6]]
                demo[info] = [year]
                #lis.append(demo)
                self.data3[item[0]] = demo
        f.flush()
        f.close()

    def readAccount(self, fileName):
        f = open(fileName, 'r')
        attri = f.readline().strip().split(',')
        attr = []
        for i in attri:
            attr.append(i)
        self.accountToID = {}   
        self.account = {}       
        self.changeAccount = set()  
        for i in f:
            item = i.strip().split(',')
            if item[1] in self.accountToID:
                if item[0] not in self.accountToID[item[1]]:
                    self.accountToID[item[1]].append(item[0])
            else:
                idd = []
                idd.append(item[0])
                self.accountToID[item[1]] = idd
            if item[0] in self.account:
                year = [item[2], item[3]]
                if item[1] in self.account[item[0]]:
                    overlap = False
                    for year_range in self.account[item[0]][item[1]]:
                        y1 = year_range[0]
                        y2 = year_range[1]
                        y3 = item[2]
                        y4 = item[3]
                        if y1 == '':
                            y1 = 0
                        else:
                            y1 = int(y1)
                        if y2 == '':
                            y2 = datetime.datetime.now().year
                        else:
                            y2 = int(y2)
                        if y3 == '':
                            y3 = 0
                        else:
                            y3 = int(y3)
                        if y4 == '':
                            y4 = datetime.datetime.now().year
                        else:
                            y4 = int(y4)
                        #   check if two periods overlap
                        if y2 < y3-1:
                            continue
                        if y4 < y1-1:
                            continue
                        overlap = True
                        mi = min([y1, y2, y3, y4])
                        ma = max([y1, y2, y3, y4])
                        if mi == 0 and ma == datetime.datetime.now().year:
                            #year = ('', '')
                            #self.account[item[0]][item[1]] = year
                            year_range[0] = ''
                            year_range[1] = ''
                        elif mi != 0 and ma == datetime.datetime.now().year:
                            #year = (mi, '')
                            #self.account[item[0]][item[1]] = year
                            year_range[0] = mi
                            year_range[1] = ''
                        elif ma != datetime.datetime.now().year and mi == 0:
                            #year = ('', ma)
                            #self.account[item[0]][item[1]] = year
                            year_range = ''
                            year_range = ma
                        else:
                            #year = (mi, ma)
                            #self.account[item[0]][item[1]] = year
                            year_range[0] = mi
                            year_range[1] = ma
                    if not overlap:
                        self.account[item[0]][item[1]].append(year)
                else:
                    self.account[item[0]][item[1]] = [year]
                    self.changeAccount.add(item[0])
            else:
                acc = {}
                year = [item[2], item[3]]
                acc[item[1]] = [year]
                self.account[item[0]] = acc

            if item[0] in self.data4:
                accounts = (item[1])
                year = [item[2], item[3]]
                if accounts in self.data4[item[0]]:
                    overlap = False
                    for year_range in self.data4[item[0]][accounts]:
                        y1 = year_range[0]
                        y2 = year_range[1]
                        y3 = item[2]
                        y4 = item[3]
                        if y1 =='':
                            y1 = 0
                        else:
                            y1 = int(y1)
                        if y2 =='':
                            y2 = datetime.datetime.now().year
                        else:
                            y2 = int(y2)
                        if y3 == '':
                            y3 = 0
                        else:
                            y3 = int(y3)
                        if y4 == '':
                            y4 = datetime.datetime.now().year
                        else:
                            y4 = int(y4)
                         #   check if two periods overlap
                        if y2 < y3-1:
                            continue
                        if y4 < y1-1:
                            continue
                        overlap = True
                        mi = min([y1, y2, y3, y4])
                        ma = max([y1, y2, y3, y4])
                        if mi == 0 and ma == datetime.datetime.now().year:
                            #year = ('', '')
                            #self.data4[item[0]][0][accounts] = year
                            year_range[0] = ''
                            year_range[1] = ''
                        elif mi != 0 and ma == datetime.datetime.now().year:
                            #year = (mi, '')
                            #self.data4[item[0]][0][accounts] = year
                            year_range[0] = mi
                            year_range[1] = ''
                        elif ma != datetime.datetime.now().year and mi == 0:
                            #year = ('', ma)
                            #self.data4[item[0]][0][accounts] = year
                            year_range[0] = ''
                            year_range[1] = ma
                        else:
                            #year = (mi, ma)
                            #self.data4[item[0]][0][accounts] = year
                            year_range[0] = mi
                            year_range[1] = ma
                    if not overlap:
                        self.data4[item[0]][accounts].append(year)
                else:
                    self.data4[item[0]][accounts] = [year]
            else:
                #lis = []
                account = {}
                accounts = (item[1])
                year = [item[2], item[3]]
                account[accounts] = [year]
                #lis.append(account)
                self.data4[item[0]] = account
        f.flush()
        f.close()

    def r(self, file):
        f = open(file, 'r')
        f.readline()
        dic = {}
        for i in f:
            m = i.strip().split(',')
            if m[0] not in dic:
                dic[m[0]] = 0
        print('length', len(dic))
        f.close()
