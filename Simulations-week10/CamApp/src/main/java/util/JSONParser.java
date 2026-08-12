package util;

import com.google.gson.*;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class JSONParser {

    private static final Gson gson = new Gson();

    /** Append one compact record per line; the runner finalizes arrays after success. */
    public static synchronized void parseAndWriteJson(String jsonString, String outputFilePath) {
        try {
            File file = new File(outputFilePath);
            JsonElement newObject = JsonParser.parseString(jsonString);
            try (FileWriter writer = new FileWriter(file, true)) {
                gson.toJson(newObject, writer);
                writer.write(System.lineSeparator());
            }

        } catch (IOException e) {
            System.err.println("Fehler beim Schreiben der Datei: " + e.getMessage());
            e.printStackTrace();
        } catch (Exception e) {
            System.err.println("Fehler beim Verarbeiten des JSON-Strings: " + e.getMessage());
            e.printStackTrace();
        }
    }

    public static synchronized void writePseudonymCount(
            String outputFilePath, String vehicleId, long changeCount) {
        File file = new File(outputFilePath);
        JsonObject counts = new JsonObject();

        try {
            if (file.exists()) {
                try (FileReader reader = new FileReader(file)) {
                    JsonElement element = JsonParser.parseReader(reader);
                    if (element.isJsonObject()) {
                        counts = element.getAsJsonObject();
                    }
                }
            }

            counts.addProperty(vehicleId, changeCount);
            try (FileWriter writer = new FileWriter(file)) {
                gson.toJson(counts, writer);
            }
        } catch (IOException e) {
            System.err.println("Error writing pseudonym debug file: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
