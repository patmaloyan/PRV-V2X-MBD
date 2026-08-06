import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.eclipse.mosaic.lib.database.Database;
import org.eclipse.mosaic.lib.geo.CartesianPoint;
import org.eclipse.mosaic.lib.geo.GeoPoint;
import org.eclipse.mosaic.lib.transform.Wgs84Projection;

import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamConstants;
import javax.xml.stream.XMLStreamReader;
import java.io.BufferedReader;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Imports the building footprints used by the active InTAS SUMO configuration
 * into the MOSAIC scenario database.
 */
public final class ImportBuildings {

    private static final double BUILDING_HEIGHT_METERS = 10.0;
    private static final int EXPECTED_BUILDINGS = 23_345;
    private static final int EXPECTED_CORNERS = 119_871;
    private static final String KNOWN_INVALID_BUILDING = "223105088";
    private static final double COORDINATE_TOLERANCE = 1e-6;

    private ImportBuildings() {
    }

    @SuppressWarnings("deprecation")
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("Usage: ImportBuildings <scenario-directory>");
        }

        Path scenarioDirectory = Path.of(args[0]).toAbsolutePath().normalize();
        Path scenarioConfig = scenarioDirectory.resolve("scenario_config.json");
        Path sumoDirectory = scenarioDirectory.resolve("sumo");
        Path sumoConfig = sumoDirectory.resolve("sumo_config.json");
        Path databasePath = scenarioDirectory.resolve("application/InTAS.db");

        JsonObject scenarioRoot = readJson(scenarioConfig);
        JsonObject projectionConfig = scenarioRoot
                .getAsJsonObject("simulation")
                .getAsJsonObject("projection");
        JsonObject center = projectionConfig.getAsJsonObject("centerCoordinates");
        JsonObject offset = projectionConfig.getAsJsonObject("cartesianOffset");

        double centerLatitude = center.get("latitude").getAsDouble();
        double centerLongitude = center.get("longitude").getAsDouble();
        double offsetX = offset.get("x").getAsDouble();
        double offsetY = offset.get("y").getAsDouble();

        String activeSumoConfiguration = readJson(sumoConfig)
                .get("sumoConfigurationFile")
                .getAsString();
        Path sumoConfigurationPath = sumoDirectory.resolve(activeSumoConfiguration).normalize();
        SumoInputs sumoInputs = readSumoInputs(sumoConfigurationPath, sumoDirectory);
        validateProjection(sumoInputs.networkFile(), offsetX, offsetY, centerLongitude);

        Path polygonFile = findActivePolygonFile(sumoInputs.additionalFiles());
        validatePolygonProjection(polygonFile, offsetX, offsetY);

        Database existingDatabase = Database.loadFromFile(databasePath.toFile());
        Counts originalCounts = Counts.from(existingDatabase);
        int existingBuildings = existingDatabase.getBuildings().size();
        int existingCorners = countCorners(existingDatabase);

        if (existingBuildings == EXPECTED_BUILDINGS && existingCorners == EXPECTED_CORNERS) {
            System.out.printf(
                    Locale.ROOT,
                    "Building database already complete: %d buildings, %d corners.%n",
                    existingBuildings,
                    existingCorners
            );
            return;
        }
        if (existingBuildings != 0 || existingCorners != 0) {
            throw new IllegalStateException(
                    "Refusing to append to a partially populated database: "
                            + existingBuildings + " buildings, " + existingCorners + " corners"
            );
        }

        Wgs84Projection projection = new Wgs84Projection(
                GeoPoint.latLon(centerLatitude, centerLongitude),
                CartesianPoint.xy(offsetX, offsetY)
        ).useZoneOfUtmOrigin();

        Database.Builder builder = Database.Builder.loadFromFile(databasePath.toFile());
        ImportSummary summary = importPolygons(polygonFile, projection, builder);
        if (summary.importedBuildings() != EXPECTED_BUILDINGS
                || summary.importedCorners() != EXPECTED_CORNERS
                || summary.invalidBuildings() != 1) {
            throw new IllegalStateException(
                    "Unexpected polygon totals: " + summary
            );
        }

        Database updatedDatabase = builder.build(false);
        Counts updatedCounts = Counts.from(updatedDatabase);
        originalCounts.assertNetworkUnchanged(updatedCounts);
        assertBuildingCounts(updatedDatabase);

        Path temporaryDatabase = databasePath.resolveSibling("InTAS.buildings.tmp.db");
        Files.deleteIfExists(temporaryDatabase);
        try {
            updatedDatabase.saveToFile(temporaryDatabase.toString());
            Database reloadedDatabase = Database.loadFromFile(temporaryDatabase.toFile());
            originalCounts.assertNetworkUnchanged(Counts.from(reloadedDatabase));
            assertBuildingCounts(reloadedDatabase);
            replaceAtomically(temporaryDatabase, databasePath);
        } finally {
            Files.deleteIfExists(temporaryDatabase);
        }

        System.out.printf(
                Locale.ROOT,
                "Imported %d buildings with %d corners from %s at %.1f m height.%n",
                summary.importedBuildings(),
                summary.importedCorners(),
                polygonFile.getFileName(),
                BUILDING_HEIGHT_METERS
        );
    }

    private static JsonObject readJson(Path path) throws Exception {
        try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            return JsonParser.parseReader(reader).getAsJsonObject();
        }
    }

    private static SumoInputs readSumoInputs(Path sumoConfiguration, Path sumoDirectory) throws Exception {
        XMLStreamReader reader = openXml(sumoConfiguration);
        Path networkFile = null;
        List<Path> additionalFiles = new ArrayList<>();
        try {
            while (reader.hasNext()) {
                if (reader.next() != XMLStreamConstants.START_ELEMENT) {
                    continue;
                }
                if ("net-file".equals(reader.getLocalName())) {
                    networkFile = sumoDirectory.resolve(reader.getAttributeValue(null, "value")).normalize();
                } else if ("additional-files".equals(reader.getLocalName())) {
                    for (String filename : reader.getAttributeValue(null, "value").split(",")) {
                        additionalFiles.add(sumoDirectory.resolve(filename.trim()).normalize());
                    }
                }
            }
        } finally {
            reader.close();
        }
        if (networkFile == null) {
            throw new IllegalStateException("No net-file in " + sumoConfiguration);
        }
        return new SumoInputs(networkFile, additionalFiles);
    }

    private static Path findActivePolygonFile(List<Path> additionalFiles) {
        return additionalFiles.stream()
                .filter(path -> path.getFileName().toString().endsWith(".poly.xml"))
                .findFirst()
                .orElseThrow(() -> new IllegalStateException("Active SUMO configuration has no polygon file"));
    }

    private static void validateProjection(
            Path networkFile,
            double expectedOffsetX,
            double expectedOffsetY,
            double centerLongitude
    ) throws Exception {
        Location location = readLocation(networkFile);
        assertOffsets(location, expectedOffsetX, expectedOffsetY, networkFile);
        int expectedZone = (int) Math.floor((centerLongitude + 180.0) / 6.0) + 1;
        if (!location.projection().contains("+zone=" + expectedZone)) {
            throw new IllegalStateException(
                    "SUMO projection does not match scenario center UTM zone " + expectedZone
            );
        }
    }

    private static void validatePolygonProjection(
            Path polygonFile,
            double expectedOffsetX,
            double expectedOffsetY
    ) throws Exception {
        assertOffsets(readLocation(polygonFile), expectedOffsetX, expectedOffsetY, polygonFile);
    }

    private static Location readLocation(Path xmlPath) throws Exception {
        XMLStreamReader reader = openXml(xmlPath);
        try {
            while (reader.hasNext()) {
                if (reader.next() == XMLStreamConstants.START_ELEMENT
                        && "location".equals(reader.getLocalName())) {
                    String[] offset = reader.getAttributeValue(null, "netOffset").split(",");
                    return new Location(
                            Double.parseDouble(offset[0]),
                            Double.parseDouble(offset[1]),
                            reader.getAttributeValue(null, "projParameter")
                    );
                }
            }
        } finally {
            reader.close();
        }
        throw new IllegalStateException("No location element in " + xmlPath);
    }

    private static void assertOffsets(
            Location location,
            double expectedOffsetX,
            double expectedOffsetY,
            Path source
    ) {
        if (Math.abs(location.offsetX() - expectedOffsetX) > COORDINATE_TOLERANCE
                || Math.abs(location.offsetY() - expectedOffsetY) > COORDINATE_TOLERANCE) {
            throw new IllegalStateException("Projection offset mismatch in " + source);
        }
    }

    private static ImportSummary importPolygons(
            Path polygonFile,
            Wgs84Projection projection,
            Database.Builder builder
    ) throws Exception {
        int importedBuildings = 0;
        int importedCorners = 0;
        int invalidBuildings = 0;
        XMLStreamReader reader = openXml(polygonFile);
        try {
            while (reader.hasNext()) {
                if (reader.next() != XMLStreamConstants.START_ELEMENT
                        || !"poly".equals(reader.getLocalName())
                        || !"building".equals(reader.getAttributeValue(null, "type"))) {
                    continue;
                }

                String id = reader.getAttributeValue(null, "id");
                List<CartesianCoordinate> coordinates = parseShape(reader.getAttributeValue(null, "shape"));
                if (!isValidPolygon(coordinates)) {
                    invalidBuildings++;
                    System.err.println("Skipping invalid building polygon " + id);
                    if (!KNOWN_INVALID_BUILDING.equals(id)) {
                        throw new IllegalStateException("Unexpected invalid building polygon " + id);
                    }
                    continue;
                }

                GeoPoint[] corners = new GeoPoint[coordinates.size()];
                for (int index = 0; index < coordinates.size(); index++) {
                    CartesianCoordinate coordinate = coordinates.get(index);
                    corners[index] = projection.cartesianToGeographic(
                            CartesianPoint.xy(coordinate.x(), coordinate.y())
                    );
                }
                builder.addBuilding(id, id, BUILDING_HEIGHT_METERS, corners);
                importedBuildings++;
                importedCorners += corners.length;
            }
        } finally {
            reader.close();
        }
        return new ImportSummary(importedBuildings, importedCorners, invalidBuildings);
    }

    private static List<CartesianCoordinate> parseShape(String shape) {
        List<CartesianCoordinate> coordinates = new ArrayList<>();
        for (String point : shape.trim().split("\\s+")) {
            String[] components = point.split(",");
            CartesianCoordinate coordinate = new CartesianCoordinate(
                    Double.parseDouble(components[0]),
                    Double.parseDouble(components[1])
            );
            if (coordinates.isEmpty() || !coordinate.equals(coordinates.get(coordinates.size() - 1))) {
                coordinates.add(coordinate);
            }
        }
        if (coordinates.size() > 1
                && coordinates.get(0).equals(coordinates.get(coordinates.size() - 1))) {
            coordinates.remove(coordinates.size() - 1);
        }
        return coordinates;
    }

    private static boolean isValidPolygon(List<CartesianCoordinate> coordinates) {
        Set<CartesianCoordinate> uniqueCoordinates = new HashSet<>(coordinates);
        if (uniqueCoordinates.size() < 3) {
            return false;
        }
        double doubledArea = 0.0;
        for (int index = 0; index < coordinates.size(); index++) {
            CartesianCoordinate current = coordinates.get(index);
            CartesianCoordinate next = coordinates.get((index + 1) % coordinates.size());
            doubledArea += current.x() * next.y() - next.x() * current.y();
        }
        return Math.abs(doubledArea) > COORDINATE_TOLERANCE;
    }

    private static XMLStreamReader openXml(Path path) throws Exception {
        InputStream input = Files.newInputStream(path);
        return XMLInputFactory.newFactory().createXMLStreamReader(input);
    }

    private static int countCorners(Database database) {
        return database.getBuildings().stream()
                .mapToInt(building -> building.getWalls().size())
                .sum();
    }

    private static void assertBuildingCounts(Database database) {
        int buildings = database.getBuildings().size();
        int corners = countCorners(database);
        if (buildings != EXPECTED_BUILDINGS || corners != EXPECTED_CORNERS) {
            throw new IllegalStateException(
                    "Database validation failed: " + buildings + " buildings, " + corners + " corners"
            );
        }
    }

    private static void replaceAtomically(Path source, Path destination) throws Exception {
        try {
            Files.move(
                    source,
                    destination,
                    StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING
            );
        } catch (AtomicMoveNotSupportedException exception) {
            Files.move(source, destination, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private record SumoInputs(Path networkFile, List<Path> additionalFiles) {
    }

    private record Location(double offsetX, double offsetY, String projection) {
    }

    private record CartesianCoordinate(double x, double y) {
    }

    private record ImportSummary(int importedBuildings, int importedCorners, int invalidBuildings) {
    }

    private record Counts(int nodes, int ways, int connections, int routes, int restrictions) {

        private static Counts from(Database database) {
            return new Counts(
                    database.getNodes().size(),
                    database.getWays().size(),
                    database.getConnections().size(),
                    database.getRoutes().size(),
                    database.getRestrictions().size()
            );
        }

        private void assertNetworkUnchanged(Counts other) {
            if (!equals(other)) {
                throw new IllegalStateException(
                        "Network data changed during building import: before=" + this + ", after=" + other
                );
            }
        }
    }
}
