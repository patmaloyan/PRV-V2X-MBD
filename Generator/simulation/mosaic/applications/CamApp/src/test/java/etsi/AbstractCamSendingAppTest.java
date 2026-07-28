package etsi;

import entities.ConfigSettings;
import org.eclipse.mosaic.fed.application.ambassador.simulation.communication.CamBuilder;
import org.eclipse.mosaic.fed.application.ambassador.simulation.communication.ReceivedAcknowledgement;
import org.eclipse.mosaic.fed.application.ambassador.simulation.communication.ReceivedV2xMessage;
import org.eclipse.mosaic.fed.application.app.api.os.VehicleOperatingSystem;
import org.eclipse.mosaic.interactions.communication.V2xMessageTransmission;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class AbstractCamSendingAppTest {

    private final TestApp app = new TestApp();

    @Test
    void speedDecreaseTriggersCam() throws Exception {
        AbstractCamSendingApp.Data oldData = data(10.0, 20.0);
        AbstractCamSendingApp.Data newData = data(10.0, 19.0);
        ConfigSettings config = isolatedConfig();
        config.velocityChange = 0.5;

        Object delta = checkMaxDelta(oldData, newData, config);

        var reasonField = delta.getClass().getDeclaredField("reason");
        reasonField.setAccessible(true);
        assertEquals("Velocity changed", reasonField.get(delta).toString());
    }

    @Test
    void headingWrapAroundUsesShortestAngularDistance() throws Exception {
        AbstractCamSendingApp.Data oldData = data(359.0, 20.0);
        AbstractCamSendingApp.Data newData = data(1.0, 20.0);
        ConfigSettings config = isolatedConfig();
        config.headingChange = 4.0;

        assertNull(checkMaxDelta(oldData, newData, config));
    }

    private Object checkMaxDelta(
            AbstractCamSendingApp.Data oldData,
            AbstractCamSendingApp.Data newData,
            ConfigSettings config
    ) throws Exception {
        Method method = AbstractCamSendingApp.class.getDeclaredMethod(
                "checkMaxDelta",
                AbstractCamSendingApp.Data.class,
                AbstractCamSendingApp.Data.class,
                ConfigSettings.class
        );
        method.setAccessible(true);
        return method.invoke(app, oldData, newData, config);
    }

    private ConfigSettings isolatedConfig() {
        ConfigSettings config = new ConfigSettings();
        config.maxInterval = null;
        config.headingChange = null;
        config.velocityChange = null;
        config.positionChange = null;
        return config;
    }

    private AbstractCamSendingApp.Data data(double heading, double velocity) {
        AbstractCamSendingApp.Data data = new AbstractCamSendingApp.Data();
        data.heading = heading;
        data.velocity = velocity;
        return data;
    }

    private static class TestApp extends AbstractCamSendingApp<VehicleOperatingSystem> {
        @Override
        public Data generateEtsiData() {
            return null;
        }

        @Override
        public boolean isInSimulationArea() {
            return true;
        }

        @Override
        public boolean isInSimulationTime() {
            return true;
        }

        @Override
        public void onAcknowledgementReceived(ReceivedAcknowledgement acknowledgement) {
        }

        @Override
        public void onCamBuilding(CamBuilder camBuilder) {
        }

        @Override
        public void onMessageReceived(ReceivedV2xMessage message) {
        }

        @Override
        public void onMessageTransmitted(V2xMessageTransmission transmission) {
        }
    }
}
