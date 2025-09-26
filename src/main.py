import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# 1. GRUNDLAGEN PyTorch
print("=== PyTorch Grundlagen ===")

# Tensor erstellen (wie numpy array, aber für GPU)
tensor = torch.tensor([1, 2, 3, 4])
print(f"Einfacher Tensor: {tensor}")

# 2D Tensor (Matrix) - für Labyrinth-Daten
maze_matrix = torch.tensor([[0, 1, 0, 0],
                           [0, 1, 0, 1], 
                           [0, 0, 0, 1],
                           [1, 1, 0, 0]], dtype=torch.float32)
print(f"Labyrinth als Tensor:\n{maze_matrix}")

# GPU verwenden (falls verfügbar)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwendetes Gerät: {device}")

# 2. NEURONALES NETZ für Labyrinth-Löser
class MazeSolverNet(nn.Module):
    def __init__(self, maze_size=10):
        super(MazeSolverNet, self).__init__()
        
        # Convolutional Layers - erkennen Muster im Labyrinth
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Fully Connected Layers - Entscheidungen treffen
        self.fc1 = nn.Linear(64 * maze_size * maze_size, 128)
        self.fc2 = nn.Linear(128, 4)  # 4 Richtungen: oben, unten, links, rechts
        
        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Eingabe: Labyrinth als Bild
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        
        # Flatten für fully connected layer
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return self.softmax(x)  # Wahrscheinlichkeiten für jede Richtung

# 3. MODELL ERSTELLEN UND TESTEN
print("\n=== Neuronales Netz ===")
model = MazeSolverNet(maze_size=4)
print(f"Modell erstellt: {model}")

# Test mit unserem Labyrinth
# Batch dimension und Channel dimension hinzufügen
maze_input = maze_matrix.unsqueeze(0).unsqueeze(0)  # (1, 1, 4, 4)
print(f"Input Shape: {maze_input.shape}")

# Vorhersage machen
with torch.no_grad():
    prediction = model(maze_input)
    print(f"Vorhersage (Richtung): {prediction}")
    
    # Beste Richtung wählen
    best_direction = torch.argmax(prediction, dim=1)
    directions = ["Oben", "Unten", "Links", "Rechts"]
    print(f"Beste Richtung: {directions[best_direction.item()]}")

# 4. TRAINING (Beispiel)
def train_model():
    print("\n=== Training ===")
    
    # Optimizer und Loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Dummy Training Data (normalerweise aus echten Labyrinth-Lösungen)
    for epoch in range(5):
        # Fake training step
        optimizer.zero_grad()
        
        # Forward pass
        output = model(maze_input)
        target = torch.tensor([2])  # Sagen wir "Links" ist richtig
        
        # Loss berechnen
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Training starten
train_model()

# 5. SVG LABYRINTH PARSER
def parse_svg_maze(svg_file):
    """
    SVG-Datei parsen und in Tensor umwandeln
    """
    print(f"\n=== SVG Parser ===")
    
    # SVG-Datei laden
    tree = ET.parse(svg_file)
    root = tree.getroot()
    
    print(f"SVG Root Tag: {root.tag}")
    
    # Alle Pfade und Linien finden (Wände)
    walls = []
    for element in root.iter():
        if element.tag.endswith('path') or element.tag.endswith('line'):
            walls.append(element.attrib)
    
    print(f"Gefundene Wände: {len(walls)}")
    return walls

def svg_to_grid(svg_walls, grid_size=20):
    """
    SVG-Koordinaten in Grid umwandeln
    """
    # Erstelle leeres Grid (0 = frei, 1 = Wand)
    grid = torch.zeros((grid_size, grid_size), dtype=torch.float32)
    
    # Hier würden Sie die SVG-Koordinaten analysieren
    # und entsprechende Grid-Zellen als Wände markieren
    
    return grid

def solve_maze_with_ai(maze_tensor, start_pos, end_pos):
    """
    Labyrinth mit KI lösen
    """
    print(f"\n=== KI-Lösung ===")
    
    current_pos = start_pos
    path = [current_pos]
    
    for step in range(20):  # Max 20 Schritte
        # Maze als Input für KI vorbereiten
        maze_input = maze_tensor.unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            # KI-Vorhersage
            prediction = model(maze_input)
            direction = torch.argmax(prediction, dim=1).item()
            
            # Bewegung basierend auf Vorhersage
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # oben, unten, links, rechts
            dy, dx = moves[direction]
            
            new_y = current_pos[0] + dy
            new_x = current_pos[1] + dx
            
            # Überprüfen ob Bewegung gültig
            if (0 <= new_y < maze_tensor.shape[0] and 
                0 <= new_x < maze_tensor.shape[1] and
                maze_tensor[new_y, new_x] == 0):  # Keine Wand
                
                current_pos = (new_y, new_x)
                path.append(current_pos)
                
                # Ziel erreicht?
                if current_pos == end_pos:
                    print(f"Ziel erreicht in {step + 1} Schritten!")
                    break
    
    return path

# BEISPIEL VERWENDUNG
if __name__ == "__main__":
    print("=== Maze AI Demo ===")
    
    # Test mit unserem kleinen Labyrinth
    start = (0, 0)
    end = (3, 3)
    
    print(f"Start: {start}, Ziel: {end}")
    path = solve_maze_with_ai(maze_matrix, start, end)
    print(f"Gefundener Pfad: {path}")
    
    # =================================================================
    # KI SPEICHERN UND LADEN
    # =================================================================
    
    print("\n=== KI SPEICHERN ===")
    
    # 1. KI-Modell speichern (nur Parameter)
    model_path = "maze_solver_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✓ Modell gespeichert unter: {model_path}")
    
    # 2. Komplettes Modell speichern (Alternative)
    complete_model_path = "complete_maze_solver.pth" 
    torch.save(model, complete_model_path)
    print(f"✓ Komplettes Modell gespeichert unter: {complete_model_path}")
    
    print("\n=== KI LADEN ===")
    
    # 3. Gespeicherte KI laden (Methode 1 - empfohlen)
    def load_trained_model(model_path, maze_size=4):
        # Neues Modell erstellen
        loaded_model = MazeSolverNet(maze_size=maze_size)
        
        # Parameter laden
        try:
            loaded_model.load_state_dict(torch.load(model_path))
            loaded_model.eval()  # Wichtig: Evaluation Mode
            print(f"✓ Modell erfolgreich geladen von: {model_path}")
            return loaded_model
        except FileNotFoundError:
            print(f"⚠ Modelldatei nicht gefunden: {model_path}")
            return None
    
    # 4. KI laden und testen
    loaded_ai = load_trained_model(model_path)
    
    if loaded_ai:
        # Test mit geladener KI
        with torch.no_grad():
            test_input = maze_matrix.unsqueeze(0).unsqueeze(0)
            prediction = loaded_ai(test_input)
            best_move = torch.argmax(prediction, dim=1).item()
            confidence = torch.max(prediction).item()
            
            directions = ["Oben", "Unten", "Links", "Rechts"]
            print(f"Geladene KI empfiehlt: {directions[best_move]} (Sicherheit: {confidence:.2%})")
    
    # SVG parsen (wenn Datei existiert)
    # try:
    #     svg_walls = parse_svg_maze("maze.svg")
    #     maze_grid = svg_to_grid(svg_walls)
    # except FileNotFoundError:
    #     print("Keine SVG-Datei gefunden")

