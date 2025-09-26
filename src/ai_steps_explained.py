# =============================================================================
# KOMPLETTE ANLEITUNG: KI für Labyrinth-Lösung erstellen und speichern
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import os

# =============================================================================
# SCHRITT 1: NEURONALES NETZ DEFINIEREN
# =============================================================================

class MazeSolverNet(nn.Module):
    """
    Unser KI-Modell für Labyrinth-Lösung
    """
    def __init__(self, maze_size=10):
        super(MazeSolverNet, self).__init__()
        
        # CONV LAYERS: Erkennen Muster im Labyrinth (Wände, Wege)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Input: 1 Kanal, Output: 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # Input: 32, Output: 64
        
        # FULLY CONNECTED: Entscheidungen treffen
        self.fc1 = nn.Linear(64 * maze_size * maze_size, 128)
        self.fc2 = nn.Linear(128, 4)  # 4 Ausgänge: Oben(0), Unten(1), Links(2), Rechts(3)
        
        # AKTIVIERUNGSFUNKTIONEN
        self.relu = nn.ReLU()           # Für versteckte Layer
        self.softmax = nn.Softmax(dim=1) # Für Wahrscheinlichkeiten
        
    def forward(self, x):
        """
        Forward Pass: Wie die KI denkt
        Input: Labyrinth als Tensor (Batch, Channel, Height, Width)
        Output: Wahrscheinlichkeiten für 4 Richtungen
        """
        # FEATURE EXTRACTION (Muster erkennen)
        x = self.relu(self.conv1(x))  # Erste Schicht
        x = self.relu(self.conv2(x))  # Zweite Schicht
        
        # FLACH MACHEN für vollverbundene Schicht
        x = x.view(x.size(0), -1)
        
        # ENTSCHEIDUNG TREFFEN
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return self.softmax(x)  # Wahrscheinlichkeiten: [0.1, 0.3, 0.5, 0.1] = Links am wahrscheinlichsten

# =============================================================================
# SCHRITT 2: TRAINING DATA VORBEREITEN
# =============================================================================

def create_training_data():
    """
    Trainingsdaten erstellen: Labyrinth + richtige Bewegung
    """
    training_data = []
    
    # BEISPIEL 1: Einfaches Labyrinth
    maze1 = torch.tensor([[0, 1, 0, 0],   # 0 = frei, 1 = Wand
                         [0, 1, 0, 1], 
                         [0, 0, 0, 1],
                         [1, 1, 0, 0]], dtype=torch.float32)
    
    # Bei Position (0,0) sollte die KI nach unten gehen (Richtung 1)
    correct_move = 1  # Unten
    training_data.append((maze1, correct_move))
    
    # BEISPIEL 2: Anderes Labyrinth
    maze2 = torch.tensor([[0, 0, 1, 0],
                         [1, 0, 1, 0], 
                         [1, 0, 0, 0],
                         [1, 1, 1, 0]], dtype=torch.float32)
    
    # Bei diesem Labyrinth sollte die KI nach rechts gehen (Richtung 3)
    correct_move = 3  # Rechts
    training_data.append((maze2, correct_move))
    
    return training_data

# =============================================================================
# SCHRITT 3: KI TRAINIEREN
# =============================================================================

def train_maze_ai(model, training_data, epochs=100):
    """
    KI trainieren: Beibringen welche Bewegungen richtig sind
    """
    print("=== TRAINING STARTET ===")
    
    # OPTIMIZER: Wie die KI lernt (Adam ist sehr gut)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # LOSS FUNCTION: Misst wie schlecht die KI gerade ist
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for maze, correct_move in training_data:
            # SCHRITT 1: Gradients zurücksetzen
            optimizer.zero_grad()
            
            # SCHRITT 2: Input vorbereiten (Batch + Channel Dimension)
            maze_input = maze.unsqueeze(0).unsqueeze(0)  # (1, 1, 4, 4)
            
            # SCHRITT 3: KI-Vorhersage
            prediction = model(maze_input)
            
            # SCHRITT 4: Target vorbereiten
            target = torch.tensor([correct_move])
            
            # SCHRITT 5: Fehler berechnen
            loss = criterion(prediction, target)
            
            # SCHRITT 6: Rückwärts lernen (Backpropagation)
            loss.backward()
            
            # SCHRITT 7: Parameter aktualisieren
            optimizer.step()
            
            total_loss += loss.item()
        
        # FORTSCHRITT ANZEIGEN
        if epoch % 20 == 0:
            avg_loss = total_loss / len(training_data)
            print(f"Epoch {epoch}, Durchschnittlicher Loss: {avg_loss:.4f}")
    
    print("=== TRAINING BEENDET ===")
    return model

# =============================================================================
# SCHRITT 4: KI SPEICHERN UND LADEN
# =============================================================================

def save_ai_model(model, filepath="maze_ai_model.pth"):
    """
    KI-Modell speichern
    """
    print(f"Speichere KI-Modell unter: {filepath}")
    
    # METHODE 1: Nur die Parameter speichern (empfohlen)
    torch.save(model.state_dict(), filepath)
    print("✓ Modell gespeichert!")

def load_ai_model(filepath="maze_ai_model.pth", maze_size=4):
    """
    KI-Modell laden
    """
    print(f"Lade KI-Modell von: {filepath}")
    
    # SCHRITT 1: Leeres Modell erstellen
    model = MazeSolverNet(maze_size=maze_size)
    
    # SCHRITT 2: Gespeicherte Parameter laden
    if os.path.exists(filepath):
        model.load_state_dict(torch.load(filepath))
        print("✓ Modell geladen!")
    else:
        print("⚠ Modelldatei nicht gefunden!")
    
    # SCHRITT 3: Evaluation Mode (wichtig!)
    model.eval()
    return model

def save_complete_model(model, filepath="complete_maze_ai.pth"):
    """
    Komplettes Modell speichern (Alternative)
    """
    torch.save(model, filepath)
    print(f"✓ Komplettes Modell gespeichert: {filepath}")

def load_complete_model(filepath="complete_maze_ai.pth"):
    """
    Komplettes Modell laden (Alternative)
    """
    if os.path.exists(filepath):
        model = torch.load(filepath)
        model.eval()
        return model
    else:
        print("⚠ Modelldatei nicht gefunden!")
        return None

# =============================================================================
# SCHRITT 5: KI VERWENDEN (INFERENZ)
# =============================================================================

def use_trained_ai(model, maze):
    """
    Trainierte KI für Vorhersagen verwenden
    """
    print("=== KI MACHT VORHERSAGE ===")
    
    # EVALUATION MODE (sehr wichtig!)
    model.eval()
    
    with torch.no_grad():  # Kein Training, nur Vorhersage
        # Input vorbereiten
        maze_input = maze.unsqueeze(0).unsqueeze(0)
        
        # KI fragen
        prediction = model(maze_input)
        
        # Beste Richtung finden
        best_direction = torch.argmax(prediction, dim=1).item()
        confidence = torch.max(prediction).item()
        
        directions = ["Oben", "Unten", "Links", "Rechts"]
        print(f"KI empfiehlt: {directions[best_direction]} (Sicherheit: {confidence:.2%})")
        
        return best_direction, confidence

# =============================================================================
# HAUPTPROGRAMM: ALLES ZUSAMMENFÜGEN
# =============================================================================

def main():
    print("=== MAZE AI TRAINING UND SPEICHERUNG ===\n")
    
    # SCHRITT 1: Modell erstellen
    model = MazeSolverNet(maze_size=4)
    print("✓ KI-Modell erstellt")
    
    # SCHRITT 2: Trainingsdaten erstellen
    training_data = create_training_data()
    print(f"✓ {len(training_data)} Trainingsbeispiele erstellt")
    
    # SCHRITT 3: KI trainieren
    trained_model = train_maze_ai(model, training_data, epochs=100)
    
    # SCHRITT 4: KI speichern
    save_ai_model(trained_model, "my_maze_ai.pth")
    save_complete_model(trained_model, "my_complete_maze_ai.pth")
    
    # SCHRITT 5: KI laden und testen
    loaded_model = load_ai_model("my_maze_ai.pth", maze_size=4)
    
    # SCHRITT 6: Test mit neuem Labyrinth
    test_maze = torch.tensor([[0, 1, 0, 0],
                             [0, 0, 0, 1], 
                             [1, 1, 0, 0],
                             [1, 1, 1, 0]], dtype=torch.float32)
    
    direction, confidence = use_trained_ai(loaded_model, test_maze)
    
    print(f"\n=== ZUSAMMENFASSUNG ===")
    print(f"✓ KI wurde trainiert und gespeichert")
    print(f"✓ KI wurde erfolgreich geladen")
    print(f"✓ KI macht Vorhersagen mit {confidence:.1%} Sicherheit")

if __name__ == "__main__":
    main()
