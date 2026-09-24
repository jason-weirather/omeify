/* Test helper only. Run with Java 11+ and a caller-supplied Bio-Formats JAR. */
import java.nio.file.Files;
import java.nio.file.Path;

import loci.common.DebugTools;
import loci.formats.FormatTools;
import loci.formats.in.OMETiffReader;

public class RGBReadback {
    public static void main(String[] args) throws Exception {
        DebugTools.setRootLevel("WARN");
        try (OMETiffReader reader = new OMETiffReader()) {
            reader.setFlattenedResolutions(false);
            reader.setId(args[0]);
            reader.setSeries(Integer.parseInt(args[1]));
            Path destination = Path.of(args[2]);
            for (int level = 0; level < reader.getResolutionCount(); level++) {
                reader.setResolution(level);
                if (!reader.isRGB() || reader.getRGBChannelCount() != 3
                        || reader.getPixelType() != FormatTools.UINT8
                        || reader.getImageCount() != 1) {
                    throw new IllegalStateException("Expected one uint8 RGB plane at level " + level);
                }
                int width = reader.getSizeX();
                int height = reader.getSizeY();
                int count = width * height;
                byte[] decoded = reader.openBytes(0);
                if (decoded.length != count * 3) {
                    throw new IllegalStateException("Unexpected decoded buffer size");
                }
                byte[] rgb = decoded;
                if (!reader.isInterleaved()) {
                    rgb = new byte[decoded.length];
                    for (int i = 0; i < count; i++) {
                        for (int c = 0; c < 3; c++) {
                            rgb[3 * i + c] = decoded[c * count + i];
                        }
                    }
                }
                Files.write(destination.resolve(level + "-" + height + "-" + width + ".rgb"), rgb);
            }
        }
    }
}
