import math
import random

# Cell Type:
# Input Cell
# Backfed Input Cell
# Noisy Input Cell

# Hidden Cell
# Probablistic Hidden Cell
# Spiking Hidden Cell

# Output Cell
# Match Input Output Cell

# Recurrent Cell
# Memory Cell
# Different Memory Cell

# Kernel
# Convolution or Pool

def function(ftype:str,z:float,prime=False,alpha=1):
    """
        type : "sigmoid,ELU,..."
        z : Pre-activation
        prime : True/False
        alpha : Default(1)
    Funtion :
    # Binary Step (z)
    # Linear (z, alpha)
    # Sigmoid (z)
    # Tanh (z)
    # ReLU (z)
    # Leaky-ReLU (z, alpha)
    # Parameterised-ReLU (z, alpha)
    # Exponential-Linear-Unit (z, alpha)
    """
    if ftype == "Binary-Step":
        if prime == False:
            if z < 0:
                y = 0
            else:
                y = 1
        # else: pas de deriver
    if ftype == "Linear":
        if prime == False:
            y = z*alpha
        else:
            y = alpha
    if ftype == "Sigmoid":
        if prime == False:
            y = 1/(1+math.exp(-z))
        else:
            y = (1/(1+math.exp(-z))) * (1-(1/(1+math.exp(-z))))
    if ftype == "Tanh":
        if prime == False:
            y = (math.exp(z)-math.exp(-z))/(math.exp(z)+math.exp(-z))
        else:
            y = 1 - (math.exp(z)-math.exp(-z))/(math.exp(z)+math.exp(-z))**2
    if ftype == "ReLU":
        if prime == False:
            y = max(0,z)
        else:
            if z >= 0:
                y = 1
            else:
                y = 0
    if ftype == "Leaky-ReLU":
        if prime == False:
            y = max(alpha*z, z)
        else:
            if z > 0:
                y = 1
            else:
                y = alpha
    if ftype == "Parameterised-ReLU":
        if prime == False:
            if z >= 0:
                y = z
            else:
                y = alpha*z
        else:
            if z >= 0:
                y = 1
            else:
                y = alpha
    if ftype == "Exponential-Linear-Unit":
        if prime == False:
            if z >= 0:
                y = z
            else:
                y = alpha*(math.exp(z)-1)
        else:
            if z >= 0:
                y = z
            else:
                y = alpha*(math.exp(y))
    return y

class neural_network:
    def __init__(self):
        self.network = [{},{},{}]
        self.used_neuron_feedforward = {}
        self.used_neuron_backward = {}
    
    def Add_Input_Neuron(self,neuron_name:str,neuron_type:str):
        """
        neuron_type:
        -Input Cell
        -Backfed Input Cell #not available
        -Noisy Input Cell #not available
        """
        self.network[0][neuron_name] = {
            "type":neuron_type,
            "output_bridge":{},
            "y":0
        }
        self.used_neuron_feedforward[neuron_name] = False
        self.used_neuron_backward[neuron_name] = False

    def Add_Hidden_Neuron(self,neuron_name:str,neuron_type:str,activation_type:str,alpha:float=None,biais:float=0.0):
        """
        neuron_type :
        -Hidden Cell
        -Probablistic Hidden Cell #not available
        -Spiking Hidden Cell #not available
        #----------#
        activation_type :
        -Binary Step (z)
        -Linear (z, alpha)
        -Sigmoid (z)
        -Tanh (z)
        -ReLU (z)
        -Leaky-ReLU (z, alpha)
        -Parameterised-ReLU (z, alpha)
        -Exponential-Linear-Unit (z, alpha)
        """
        self.network[1][neuron_name] = {
            "type":neuron_type,
            "activation":{
                "ftype":activation_type,
                "alpha":alpha
            },
            "input_bridge":{},
            "output_bridge":{},
            "biais":biais,
            "y":0,
            "delta":0
        }
        self.used_neuron_feedforward[neuron_name] = False
        self.used_neuron_backward[neuron_name] = False

    def Add_Output_Neuron(self,neuron_name:str,neuron_type:str,activation_type:str,alpha:float=None,biais:float=0.0):
        """
        neuron_type :
        -Output Cell
        -Match Input Output Cell #not available
        #----------#
        activation_type :
        -Binary Step (z)
        -Linear (z, alpha)
        -Sigmoid (z)
        -Tanh (z)
        -ReLU (z)
        -Leaky-ReLU (z, alpha)
        -Parameterised-ReLU (z, alpha)
        -Exponential-Linear-Unit (z, alpha)
        """
        self.network[2][neuron_name] = {
            "type":neuron_type,
            "activation":{
                "ftype":activation_type,
                "alpha":alpha
            },
            "input_bridge":{},
            "biais":biais,
            "y":0,
            "delta":0
        }
        self.used_neuron_feedforward[neuron_name] = False
        self.used_neuron_backward[neuron_name] = False

    def Add_Bridge(self,bridge_list:list):
        """
        bridge_list:
            [
                [from,to],
                [from,to],
                ...
            ]
        """
        #pour tout les ponts
        for bridge in bridge_list:
            #on recherche sur tout l'INPUT_LAYER
            for input_neuron in self.network[0]:
                #si un des neurone de la couche est dans le ponts selectionner
                if input_neuron in bridge:
                    #on ajoute le deuxieme neurone dans l'output
                    self.network[0][input_neuron]["output_bridge"][bridge[1]] = random.uniform(-1,1)
            #on recherche sur tout l'HIDDEN_LAYER
            for hidden_neuron in self.network[1]:
                #si un des neurone de la couche est dans le ponts selectionner
                if hidden_neuron in bridge:
                    #on verifie le type (out/in)->(0,1)
                    types = bridge.index(hidden_neuron)
                    if types == 0:#si c'est un ponts sortant
                        self.network[1][hidden_neuron]["output_bridge"][bridge[1]] = random.uniform(-1,1)
                    else:#si c'est un ponts entrant
                        self.network[1][hidden_neuron]["input_bridge"][bridge[0]] = random.uniform(-1,1)
            #on recherche sur tout l'OUTPUT_LAYER
            for output_neuron in self.network[2]:
                #si un des neurone de la couche est dans le ponts selectionner
                if output_neuron in bridge:
                    self.network[2][output_neuron]["input_bridge"][bridge[0]] = random.uniform(-1,1)

    def train(self,inputs,expected,learning_rate,nb_epoch,display=False):
        #pour chaque epoch
        for epoch in range(nb_epoch):
            #definir l'erreur de l'epoch a 0
            error = 0
            #pour tout les inputs
            for x,values in enumerate(inputs):
                #faire le feet_forward avec les valeurs
                outputs = self.feed_forward(values)
                #on fait l'addition de toute les difference de l'output
                error += sum([(expected[x][i]-outputs[i])**2 for i in range(len(expected[x]))])
                #on calcule le taux d'erreur de chaque neurone
                self.backward(expected[x])
                self.update_weights(values,learning_rate)
            if display == True:
                print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learning_rate, error))

    def feed_forward(self,inputs_values):
        ####METTRE TOUT LES NEURONES A "NON UTILISER"####
        self.used_neuron_feedforward = {x: False for x in self.used_neuron_feedforward}
        #################################################
        
        #pour chaque neurone de l'INPUT_LAYER
        for x_input_neuron,input_neuron in enumerate(self.network[0]):
            #definir la valeur des inputs
            self.network[0][input_neuron]["y"] = inputs_values[x_input_neuron]
            self.used_neuron_feedforward[input_neuron] = True
        
        #tant que tout les neurones ne sont pas calculer
        while all(self.used_neuron_feedforward[x] == True for x in self.used_neuron_feedforward) == False:
            #pour chaque couche de l'HIDDEN_LAYER
            for hidden_neuron in self.network[1]:
                #si tout les neurones d'entré sont calculer
                if all(self.used_neuron_feedforward[in_neuron]==True for in_neuron in self.network[1][hidden_neuron]["input_bridge"]):
                    #preactivation du neurone
                    z = self.pre_activation(self.network[1][hidden_neuron])
                    #on calcul l'activation
                    y = function(self.network[1][hidden_neuron]["activation"]["ftype"],z,alpha=self.network[1][hidden_neuron]["activation"]["alpha"])
                    self.network[1][hidden_neuron]["y"] = y
                    self.used_neuron_feedforward[hidden_neuron] = True
            #pour chaque couche de l'OUTPUT_LAYER
            for output_neuron in self.network[2]:
                #si tout les neurones d'entré sont calculer
                if all(self.used_neuron_feedforward[in_neuron]==True for in_neuron in self.network[2][output_neuron]["input_bridge"]):
                    #preactivation du neurone
                    z = self.pre_activation(self.network[2][output_neuron])
                    #on calcul l'activation
                    y = function(self.network[2][output_neuron]["activation"]["ftype"],z,alpha=self.network[2][output_neuron]["activation"]["alpha"])
                    self.network[2][output_neuron]["y"] = y
                    self.used_neuron_feedforward[output_neuron] = True
        outputs = [self.network[2][x]["y"] for x in self.network[2]]
        return outputs

    def pre_activation(self,current_neuron):
        z = current_neuron["biais"]
        #pour tout les neurones entrant
        for in_neuron in current_neuron["input_bridge"]:
            #calculer valeur * poids
            #chercher couche par couche le neurone demander
            for layer in self.network:
                #si la couche contient le neurone
                if in_neuron in layer.keys():
                    in_neuron_data = layer[in_neuron]
                    break
            z += in_neuron_data["y"]*current_neuron["input_bridge"][in_neuron]
        return z

    def backward(self,expected):
        ####METTRE TOUT LES NEURONES A "NON UTILISER"####
        self.used_neuron_backward = {x: False for x in self.used_neuron_backward}
        #on met tout les neurone de l'INPUT_LAYER a True
        for input_neuron in self.network[0]:
            self.used_neuron_backward[input_neuron] = True
        #################################################
        #Calcul de l'erreur de l'OUTPUT_LAYER
        #pour chaque neurones de l'OUTPUT_LAYER
        for x_output_neuron,output_neuron in enumerate(self.network[2]):
            #on calcule la difference entre l'attendue et ce qu'on a eu
            error = expected[x_output_neuron] - self.network[2][output_neuron]["y"]
            #on calcul le taux d'erreur de ce neurone
            self.network[2][output_neuron]['delta'] = error* function(self.network[2][output_neuron]["activation"]["ftype"],self.network[2][output_neuron]["y"],prime=True,alpha=self.network[2][output_neuron]["activation"]["alpha"])
            self.used_neuron_backward[output_neuron] = True

        #Tant que tout les hidden
        while all(self.used_neuron_backward[x] == True for x in self.used_neuron_backward) == False:
            #pour chaque couche de l'HIDDEN_LAYER
            for hidden_neuron in self.network[1]:
                #si tout les neurones de sortie sont calculer
                if all(self.used_neuron_backward[out_neuron]==True for out_neuron in self.network[1][hidden_neuron]["output_bridge"]) and not all(self.used_neuron_backward[n] == True for n in self.used_neuron_backward):
                    #definir l'erreur à 0
                    error = 0.0
                    #pour chaque neurone de sortie
                    for out_neuron in self.network[1][hidden_neuron]["output_bridge"]:
                        #si le neurones est dans l'HIDDEN_LAYER
                        if out_neuron in self.network[1].keys():
                            #multiplier le poid du pont et le taux d'erreur du neurone de sortie
                            error += (self.network[1][hidden_neuron]['output_bridge'][out_neuron] * self.network[1][out_neuron]['delta'])
                        #si le neurones est dans l'OUTPUT_LAYER
                        if out_neuron in self.network[2].keys():
                            #multiplier le poid du pont et le taux d'erreur du neurone de sortie
                            error += (self.network[1][hidden_neuron]['output_bridge'][out_neuron] * self.network[2][out_neuron]['delta'])
                    #on ajoute le taux d'erreur de ce neurone a la liste
                    self.network[1][hidden_neuron]["delta"] = error * function(self.network[1][hidden_neuron]["activation"]["ftype"],self.network[1][hidden_neuron]["y"],prime=True,alpha=self.network[1][hidden_neuron]["activation"]["alpha"])
                    self.used_neuron_backward[hidden_neuron] = True

    def update_weights(self,inputs,learning_rate):
        ####METTRE TOUT LES NEURONES A "NON UTILISER"####
        self.used_neuron_feedforward = {x: False for x in self.used_neuron_feedforward}
        #################################################
        
        #pour chaque neurone de l'INPUT_LAYER
        for x_input_neuron,input_neuron in enumerate(self.network[0]):
            #definir la valeur des inputs
            self.network[0][input_neuron]["y"] = inputs[x_input_neuron]
            self.used_neuron_feedforward[input_neuron] = True

        #Tant que tout les neurone ne sont pas calculer
        while all(self.used_neuron_feedforward[x] == True for x in self.used_neuron_feedforward) == False:
            #pour chaque neurone de l'HIDDEN_LAYER
            for hidden_neuron in self.network[1]:
                #si tout les neurones d'entré sont calculer
                if all(self.used_neuron_feedforward[in_neuron]==True for in_neuron in self.network[1][hidden_neuron]["input_bridge"]):
                    #Pour chaque entré
                    for in_neuron in self.network[1][hidden_neuron]["input_bridge"]:
                        #si l'entré est dans l'INPUT_LAYER
                        if in_neuron in self.network[0].keys():
                            #on update le poids de l'entré
                            #du neuron actuel
                            self.network[1][hidden_neuron]["input_bridge"][in_neuron] += learning_rate * self.network[1][hidden_neuron]["delta"] * self.network[0][in_neuron]["y"]
                            #de l'input neuron
                            self.network[0][in_neuron]["output_bridge"][hidden_neuron] = self.network[1][hidden_neuron]["input_bridge"][in_neuron]
                        #si l'entré est dans l'HIDDEN_LAYER
                        if in_neuron in self.network[1].keys():
                            #on update le poids de l'entré
                            #du neuron actuel
                            self.network[1][hidden_neuron]["input_bridge"][in_neuron] += learning_rate * self.network[1][hidden_neuron]["delta"] * self.network[1][in_neuron]["y"]
                            #du neuron precedent
                            self.network[1][in_neuron]["output_bridge"][hidden_neuron] = self.network[1][hidden_neuron]["input_bridge"][in_neuron]
                    #on met a jour le biais
                    self.network[1][hidden_neuron]["biais"] += learning_rate * self.network[1][hidden_neuron]["delta"]
                    self.used_neuron_feedforward[hidden_neuron] = True
            #pour chaque neurone de l'OUTPUT_LAYER
            for output_neuron in self.network[2]:
                #si tout les neurones d'entré sont calculer
                if all(self.used_neuron_feedforward[in_neuron]==True for in_neuron in self.network[2][output_neuron]["input_bridge"]):
                    #Pour chaque entré
                    for in_neuron in self.network[2][output_neuron]['input_bridge']:
                        #si l'entré est dans l'INPUT_LAYER
                        if in_neuron in self.network[0].keys():
                            #on update le poids de l'entré
                            #du neuron actuel
                            self.network[2][output_neuron]["input_bridge"][in_neuron] += learning_rate * self.network[2][output_neuron]["delta"] * self.network[0][in_neuron]["y"]
                            #de l'input neuron
                            self.network[0][in_neuron]["output_bridge"][output_neuron] = self.network[2][output_neuron]["input_bridge"][in_neuron]
                        #si l'entré est dans l'HIDDEN_LAYER
                        if in_neuron in self.network[1].keys():
                            #on update le poids de l'entré
                            #du neuron actuel
                            self.network[2][output_neuron]["input_bridge"][in_neuron] += learning_rate * self.network[2][output_neuron]["delta"] * self.network[1][in_neuron]["y"]
                            #du neuron precedent
                            self.network[1][in_neuron]["output_bridge"][output_neuron] = self.network[2][output_neuron]["input_bridge"][in_neuron]
                    #on met a jour le biais
                    self.network[2][output_neuron]["biais"] += learning_rate * self.network[2][output_neuron]["delta"]
                    self.used_neuron_feedforward[output_neuron] = True

    def predict(self,inputs):
        outputs = self.feed_forward(inputs)
        return outputs


