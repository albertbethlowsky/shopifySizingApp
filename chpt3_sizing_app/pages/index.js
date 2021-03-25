import { Heading, Page } from "@shopify/polaris";
import React, { useState } from "react";
import Modal from "react-modal";

const Index = () => {
  const [modalVisible, setModalVisible] = useState(false);

  function openModal() {
    setModalVisible(true);
  }

  function closeModal() {
    setModalVisible(false);
  }

  return (
    <div>
      <button onClick={openModal}>
        Not sure about your size? Let us help.
      </button>
      <Modal
        isOpen={modalVisible}
        onRequestClose={closeModal}
        contentLabel="Find your perfect size"
      >
        <input />
      </Modal>
    </div>
  );
};

export default Index;
