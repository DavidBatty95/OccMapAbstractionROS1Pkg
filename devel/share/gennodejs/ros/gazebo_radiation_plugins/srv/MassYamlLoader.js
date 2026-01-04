// Auto-generated. Do not edit!

// (in-package gazebo_radiation_plugins.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------


//-----------------------------------------------------------

class MassYamlLoaderRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.folder = null;
    }
    else {
      if (initObj.hasOwnProperty('folder')) {
        this.folder = initObj.folder
      }
      else {
        this.folder = '';
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type MassYamlLoaderRequest
    // Serialize message field [folder]
    bufferOffset = _serializer.string(obj.folder, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type MassYamlLoaderRequest
    let len;
    let data = new MassYamlLoaderRequest(null);
    // Deserialize message field [folder]
    data.folder = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += _getByteLength(object.folder);
    return length + 4;
  }

  static datatype() {
    // Returns string type for a service object
    return 'gazebo_radiation_plugins/MassYamlLoaderRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'bf1b291db59c399c5c5b25859bd3e2e3';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    string folder
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new MassYamlLoaderRequest(null);
    if (msg.folder !== undefined) {
      resolved.folder = msg.folder;
    }
    else {
      resolved.folder = ''
    }

    return resolved;
    }
};

class MassYamlLoaderResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
    }
    else {
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type MassYamlLoaderResponse
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type MassYamlLoaderResponse
    let len;
    let data = new MassYamlLoaderResponse(null);
    return data;
  }

  static getMessageSize(object) {
    return 0;
  }

  static datatype() {
    // Returns string type for a service object
    return 'gazebo_radiation_plugins/MassYamlLoaderResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'd41d8cd98f00b204e9800998ecf8427e';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new MassYamlLoaderResponse(null);
    return resolved;
    }
};

module.exports = {
  Request: MassYamlLoaderRequest,
  Response: MassYamlLoaderResponse,
  md5sum() { return 'bf1b291db59c399c5c5b25859bd3e2e3'; },
  datatype() { return 'gazebo_radiation_plugins/MassYamlLoader'; }
};
