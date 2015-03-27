from pgmpy.models import BayesianModel
from pgmpy.factors import TabularCPD


class BIFReader:

    def __init__(self, path):
        self.path = path
        self.bifFile = open(path)

    def get_bayesian_model(self):
        # Control flags
        getVarFlag = False
        getCptFlag = False

        # Temp vars
        tempVar = {"name": "", "domain": []}    # Ex: {"name": "var1", "domain": ["true","false"]}
        tempVars = {}   # Ex: {"var1": {"name": "var1", "domain": ["true","false"]}}
        tempCpt = {"head": [], "tail": [], "values": []}    # Ex: {"head": ["var1"], "tail": ["var3","var4"], "values": [0.1 0.9 0.2 0.8]}
        tempCpts = {}   # Ex: {"var1": {"head": ["var1"], "tail": ["var3","var4"]}}

        for line in self.bifFile:

            # Construct the DAG
            if getCptFlag or line.find("probability") == 0:
                if not getCptFlag:
                    splitLine = line.replace(",", "").split()
                    head = []
                    tail = []
                    headFlag = False
                    tailFlag = False
                    for term in splitLine:
                        if headFlag:
                            head.append(term)
                        elif tailFlag:
                            tail.append(term)

                        if term == "(":
                            headFlag = True
                        elif term == "|":
                            head.pop()
                            headFlag = False
                            tailFlag = True
                        elif term == ")":
                            if len(tail) != 0:
                                tail.pop()
                            else:
                                head.pop()
                            headFlag = False
                            tailFlag = False

                    # add edge in BN
                    # if len(tail) != 0:
                    #     for parent in tail:
                    #         for child in head:
                    #             self.bn.add_edge(tempVars[parent]["name"], tempVars[child]["name"])

                    # Save head and tail to temporary CPT
                    for headVar in head:
                        tempCpt["head"].append(tempVars[headVar]["name"])
                    for tailVar in tail:
                        tempCpt["tail"].append(tempVars[tailVar]["name"])

                    getCptFlag = True

                else:
                    # Construct the table while } is not found
                    if line.find("}") == -1:
                        splitLine = line.strip()
                        domainSplit = []
                        probabilitiesSplit = []

                        if splitLine.find("table") == 0:
                            probabilitiesSplit = splitLine.replace(
                                "table ", "").replace(";", "").split(",")
                        else:
                            domainSplit = splitLine.split(")")[0].replace(
                                "(", "").split(",")
                            domainSplit = [s.strip() for s in domainSplit]
                            probabilitiesSplit = splitLine.split(")")[1].replace(
                                "(", "").replace(";", "").split(",")

                        probabilitiesSplit = [
                            float(s.strip()) for s in probabilitiesSplit]

                        for headVar in tempCpt["head"]:
                            if len(tempCpt["values"]) == 0:
                                tempCpt["values"] = [[] for _ in range(0, len(tempVars[headVar]["domain"]))]
                            for counter, domainValue in enumerate(tempVars[headVar]["domain"]):
                                # tableKey = [domainValue] + domainSplit
                                tableValue = probabilitiesSplit[counter]
                                tempCpt["values"][counter].append(tableValue)
                    else:
                        tempCpts[tempCpt["head"][0]] = tempCpt.copy()
                        tempCpt = {"head": [], "tail": [], "values": []}
                        getCptFlag = False

            # Construct domain of Variables
            elif getVarFlag or line.find("variable") == 0:
                splitLine = line.split()
                if not getVarFlag:
                    varName = ""
                    controlFlag = False
                    for term in splitLine:
                        if controlFlag:
                            varName = varName + term

                        if term == "variable":
                            controlFlag = True
                        elif term == "{":
                            varName = varName.replace("{", "")
                            controlFlag = False
                    tempVar["name"] = varName
                    getVarFlag = True
                else:
                    if line.find("  type discrete") == 0:
                        controlFlag = False
                        for term in splitLine:
                            if controlFlag:
                                tempVar["domain"].append(term.replace(",", ""))
                            if term == "{":
                                controlFlag = True
                            elif term == "};":
                                tempVar["domain"].remove("};")
                                controlFlag = False
                        tempVars[tempVar["name"]] = tempVar.copy()
                        tempVar = {"name": "", "domain": []}
                        getVarFlag = False
                    else:
                        print("ERROR! Not discrete variable!")
        # Construct the Bayesian Model
        BN = BayesianModel()
        all_nodes = [v for v in tempVars]
        BN.add_nodes_from(all_nodes)
        for head_var in tempCpts:
            # Adding edges
            all_edges = [(t, head_var) for t in tempCpts[head_var]["tail"]]
            for edge in all_edges:
                BN.add_edge(edge[0], edge[1])
            # Adding CPDs
            head_card = len(tempVars[head_var]["domain"])
            evidence_card = [len(tempVars[v]["domain"]) for v in tempCpts[head_var]["tail"]]
            newCPD = TabularCPD(head_var,
                                head_card,
                                tempCpts[head_var]["values"],
                                evidence=tempCpts[head_var]["tail"],
                                evidence_card=evidence_card)
            BN.add_cpds(newCPD)
        self.bifFile.close()
        return BN
