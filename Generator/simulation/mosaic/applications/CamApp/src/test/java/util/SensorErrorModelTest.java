package util;

import org.eclipse.mosaic.lib.geo.MutableCartesianPoint;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SensorErrorModelTest {

    @Test
    void reportsEveryNoiseValueAsReportedMinusTrue() {
        SensorErrorModel model = new SensorErrorModel(new Random(42L));

        MutableCartesianPoint truePosition = new MutableCartesianPoint(100.0, 200.0, 3.0);
        Pair<MutableCartesianPoint, MutableCartesianPoint> position = model.addPositionNoise(truePosition);
        Pair<Double, Double> speed = model.addSpeedNoise(20.0);
        Pair<Double, Double> heading = model.addHeadingNoise(355.0, 20.0);

        assertEquals(
                position.getFirst().getX() - truePosition.getX(),
                position.getSecond().getX(),
                1e-12
        );
        assertEquals(
                position.getFirst().getY() - truePosition.getY(),
                position.getSecond().getY(),
                1e-12
        );
        assertEquals(3.0, position.getFirst().getZ(), 0.0);
        assertEquals(speed.getFirst() - 20.0, speed.getSecond(), 1e-12);

        double circularHeadingError = heading.getFirst() - 355.0;
        if (circularHeadingError > 180.0) {
            circularHeadingError -= 360.0;
        } else if (circularHeadingError < -180.0) {
            circularHeadingError += 360.0;
        }
        assertEquals(circularHeadingError, heading.getSecond(), 1e-12);
    }

    @Test
    void accelerationIsFiniteForFirstAndRepeatedTimestamps() {
        SensorErrorModel model = new SensorErrorModel(new Random(7L));

        model.addSpeedNoise(10.0);
        Pair<Double, Double> first = model.addAccelerationNoise(1.5, 1_000_000_000L);

        model.addSpeedNoise(11.0);
        Pair<Double, Double> repeatedTimestamp =
                model.addAccelerationNoise(1.5, 1_000_000_000L);

        model.addSpeedNoise(12.0);
        Pair<Double, Double> later =
                model.addAccelerationNoise(1.5, 2_000_000_000L);

        assertEquals(1.5, first.getFirst(), 0.0);
        assertEquals(0.0, first.getSecond(), 0.0);
        assertEquals(1.5, repeatedTimestamp.getFirst(), 0.0);
        assertEquals(0.0, repeatedTimestamp.getSecond(), 0.0);
        assertTrue(Double.isFinite(later.getFirst()));
        assertTrue(Double.isFinite(later.getSecond()));
        assertEquals(later.getFirst() - 1.5, later.getSecond(), 1e-12);
    }
}
