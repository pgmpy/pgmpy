from pgmpy.models import BayesianModel
from pgmpy.factors import TabularCPD


class BIFReader(object):

    def __init__(self, path):
        self.path = path
        self.bif_file = open(path)

    def get_bayesian_model(self):
        # Control flags
        get_var_flag = False
        get_cpt_flag = False

        # Temp vars
        # Ex: {"name": "var1", "domain": ["true","false"]}
        temp_var = {"name": "", "domain": []}
        # Ex: {"var1": {"name": "var1", "domain": ["true","false"]}}
        temp_vars = {}
        # Ex: {"head": ["var1"], "tail": ["var3","var4"], "values": [0.1 0.9
        # 0.2 0.8]}
        temp_cpt = {"head": [], "tail": [], "values": []}
        # Ex: {"var1": {"head": ["var1"], "tail": ["var3","var4"]}}
        temp_cpts = {}

        for line in self.bif_file:

            # Construct the DAG
            if get_cpt_flag or line.find("probability") == 0:
                if not get_cpt_flag:
                    split_line = line.replace(",", "").split()
                    head = []
                    tail = []
                    head_flag = False
                    tail_flag = False
                    for term in split_line:
                        if head_flag:
                            head.append(term)
                        elif tail_flag:
                            tail.append(term)

                        if term == "(":
                            head_flag = True
                        elif term == "|":
                            head.pop()
                            head_flag = False
                            tail_flag = True
                        elif term == ")":
                            if len(tail) != 0:
                                tail.pop()
                            else:
                                head.pop()
                            head_flag = False
                            tail_flag = False

                    # Save head and tail to temporary CPT
                    for head_var in head:
                        temp_cpt["head"].append(temp_vars[head_var]["name"])
                    for tail_var in tail:
                        temp_cpt["tail"].append(temp_vars[tail_var]["name"])

                    get_cpt_flag = True

                else:
                    # Construct the table while } is not found
                    if line.find("}") == -1:
                        split_line = line.strip()
                        domain_split = []
                        probabilities_split = []

                        if split_line.find("table") == 0:
                            probabilities_split = split_line.replace(
                                "table ", "").replace(";", "").split(",")
                        else:
                            domain_split = split_line.split(")")[0].replace(
                                "(", "").split(",")
                            domain_split = [s.strip() for s in domain_split]
                            probabilities_split = split_line.split(")")[1].replace(
                                "(", "").replace(";", "").split(",")

                        probabilities_split = [
                            float(s.strip()) for s in probabilities_split]

                        for head_var in temp_cpt["head"]:
                            if len(temp_cpt["values"]) == 0:
                                temp_cpt["values"] = [
                                    [] for _ in range(0, len(temp_vars[head_var]["domain"]))]
                            for counter, _ in enumerate(temp_vars[head_var]["domain"]):
                                table_value = probabilities_split[counter]
                                temp_cpt["values"][counter].append(table_value)
                    else:
                        temp_cpts[temp_cpt["head"][0]] = temp_cpt.copy()
                        temp_cpt = {"head": [], "tail": [], "values": []}
                        get_cpt_flag = False

            # Construct domain of Variables
            elif get_var_flag or line.find("variable") == 0:
                split_line = line.split()
                if not get_var_flag:
                    var_name = ""
                    control_flag = False
                    for term in split_line:
                        if control_flag:
                            var_name = var_name + term

                        if term == "variable":
                            control_flag = True
                        elif term == "{":
                            var_name = var_name.replace("{", "")
                            control_flag = False
                    temp_var["name"] = var_name
                    get_var_flag = True
                else:
                    if line.find("  type discrete") == 0:
                        control_flag = False
                        for term in split_line:
                            if control_flag:
                                temp_var["domain"].append(
                                    term.replace(",", ""))
                            if term == "{":
                                control_flag = True
                            elif term == "};":
                                temp_var["domain"].remove("};")
                                control_flag = False
                        temp_vars[temp_var["name"]] = temp_var.copy()
                        temp_var = {"name": "", "domain": []}
                        get_var_flag = False
                    else:
                        print("ERROR! Not discrete variable!")
        # Construct the Bayesian Model
        bayesian_model = BayesianModel()
        all_nodes = [v for v in temp_vars]
        bayesian_model.add_nodes_from(all_nodes)
        for head_var in temp_cpts:
            # Adding edges
            all_edges = [(t, head_var) for t in temp_cpts[head_var]["tail"]]
            for edge in all_edges:
                bayesian_model.add_edge(edge[0], edge[1])
            # Adding CPDs
            head_card = len(temp_vars[head_var]["domain"])
            evidence_card = [len(temp_vars[v]["domain"])
                             for v in temp_cpts[head_var]["tail"]]
            new_cpd = TabularCPD(head_var,
                                 head_card,
                                 temp_cpts[head_var]["values"],
                                 evidence=temp_cpts[head_var]["tail"],
                                 evidence_card=evidence_card)
            bayesian_model.add_cpds(new_cpd)
        self.bif_file.close()
        return bayesian_model
