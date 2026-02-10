<!DOCTYPE html>
<html>
<head>
    <title>Recomendador de Películas</title>
</head>
<body>
    <h1>Perfil del Usuario</h1>

    <form action="/predict" method="post">
        <label>Rating promedio:</label>
        <input type="number" step="0.1" name="rating" required>

        <h3>Géneros vistos:</h3>
        <input type="checkbox" name="genres" value="Action"> Action<br>
        <input type="checkbox" name="genres" value="Comedy"> Comedy<br>
        <input type="checkbox" name="genres" value="Drama"> Drama<br>
        <input type="checkbox" name="genres" value="Romance"> Romance<br>
        <input type="checkbox" name="genres" value="Sci-Fi"> Sci-Fi<br>

        <br>
        <button type="submit">Predecir Clúster</button>
    </form>
</body>
</html>
